#!/usr/bin/env python3
# /// script
# dependencies = [
#   "boto3",
#   "datasets",
#   "Pillow",
#   "transformers[torch]",
# ]
# ///
import os
# Suppress the tqdm progress bar spam in non-TTY environments
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import requests
import boto3
import torch
import time
import logging
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from botocore.config import Config

# Configure structured logging for Kubernetes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_token(ls_url, ls_refresh_token):
    token_url = f'{ls_url}/api/token/refresh'
    payload = {'refresh': ls_refresh_token}
    response = requests.post(token_url, json=payload)
    tokens = response.json()
    return tokens['access']

def with_retry(num_retries=5):
    def wrapper(fn):
        def inner(ls_url, ls_refresh_token, ls_headers, *args, **kwargs):
            for attempt in range(1, num_retries + 1):
                try:
                    res = fn(ls_headers=ls_headers, ls_url=ls_url, ls_refresh_token=ls_refresh_token, *args, **kwargs)
                    
                    if res.status_code in [200, 201]:
                        return res

                    if res.status_code == 401:
                        logger.info("Token expired. Refreshing Label Studio authentication...")
                        ls_token = get_token(ls_url, ls_refresh_token)
                        ls_headers['Authorization'] = f"Bearer {ls_token}"
                        continue

                    if 200 <= res.status_code < 500:
                        return res

                    res.raise_for_status()

                except Exception as ex:
                    logger.warning(f"API Request failed: {ex}. Retrying in {attempt * 10} seconds...")
                    time.sleep(attempt * 10)

            # Reached only when all num_retries iterations are exhausted
            logger.error(f"Maximum API retries ({num_retries}) exceeded.")
            raise RuntimeError(f"Failed to complete API request after {num_retries} attempts.")

        return inner
    return wrapper

@with_retry()
def get_task_page(tasks_url, ls_headers, ls_url, ls_refresh_token):
    return requests.get(tasks_url, headers=ls_headers)

@with_retry()
def get_predictions(pred_url, ls_headers, ls_url, ls_refresh_token):
    return requests.get(pred_url, headers=ls_headers)

@with_retry()
def delete_predictions(pred_url, ls_headers, ls_url, ls_refresh_token):
    return requests.delete(pred_url, headers=ls_headers)

@with_retry()
def create_predictions(pred_url, payload, ls_headers, ls_url, ls_refresh_token):
    return requests.post(pred_url, headers=ls_headers, json=payload)

def main():
    parser = argparse.ArgumentParser(description="Auto-label Label Studio tasks using Grounding DINO.")
    parser.add_argument("--ls-url", default="https://label-studio.local.godesteem.de", help="Label Studio URL")
    parser.add_argument("--project-id", required=True, type=int, help="Label Studio Project ID")
    parser.add_argument("--prompt", default="plant . insect .", help="Text prompt for Grounding DINO")
    parser.add_argument("--threshold", default=0.3, type=float, help="Confidence threshold")
    parser.add_argument("--bucket", default="insects", help="MinIO bucket name")
    parser.add_argument("--minio-endpoint", default="http://minio.private.svc.cluster.local:9000", help="Internal MinIO IP")
    parser.add_argument("--clear-existing", action="store_true", help="Delete existing predictions on the task before adding new ones")    
    args = parser.parse_args()

    ls_refresh_token = os.environ.get("LABEL_STUDIO_TOKEN")
    minio_ak = os.environ.get("MINIO_ACCESS_KEY")
    minio_sk = os.environ.get("MINIO_SECRET_KEY")

    if not all([ls_refresh_token, minio_ak, minio_sk]):
        logger.critical("Missing required environment variables: LABEL_STUDIO_TOKEN, MINIO_ACCESS_KEY, MINIO_SECRET_KEY")
        exit(1)

    logger.info("Loading Grounding DINO model into memory...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    logger.info(f"Connecting to MinIO at {args.minio_endpoint}...")
    s3_client = boto3.client(
        's3',
        endpoint_url=args.minio_endpoint,
        aws_access_key_id=minio_ak,
        aws_secret_access_key=minio_sk,
        region_name='us-east-1',
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}) 
    )

    logger.info(f"Fetching tasks for Project {args.project_id}...")
    ls_token = get_token(args.ls_url, ls_refresh_token)
    ls_headers = {"Authorization": f"Bearer {ls_token}"}
    
    og_tasks_url = f"{args.ls_url}/api/projects/{args.project_id}/tasks?page_size=10"
    page_no = 1
    tasks_url = f'{og_tasks_url}&page={page_no}'
    total_processed = 0

    while tasks_url:
        logger.info(f"Fetching page {page_no}[size: 10] of tasks...")
        response = get_task_page(tasks_url=tasks_url, ls_headers=ls_headers, ls_url=args.ls_url, ls_refresh_token=ls_refresh_token)
        if response.status_code != 200:
            logger.error(f"Error fetching tasks: {response.text}")
            break
            
        tasks = response.json()
        if not tasks:
            logger.info("No more tasks to process. Exiting.")
            break

        for task in tasks:
            task_id = task['id']
            image_uri = task['data'].get('image', '')
            if not image_uri.startswith('s3://'):
                continue
            
            if args.clear_existing:
                pred_url = f"{args.ls_url}/api/predictions/?task={task_id}"
                pred_resp = get_predictions(pred_url=pred_url, ls_headers=ls_headers, ls_url=args.ls_url, ls_refresh_token=ls_refresh_token)
                
                if pred_resp.status_code == 200:
                    for pred in pred_resp.json():
                        pred_id = pred.get('id')
                        if pred_id:
                            del_url = f"{args.ls_url}/api/predictions/{pred_id}/"
                            del_resp = delete_predictions(pred_url=del_url, ls_headers=ls_headers, ls_url=args.ls_url, ls_refresh_token=ls_refresh_token)
                            if del_resp.status_code not in (200, 204):
                                logger.warning(f"Failed to delete prediction {pred_id} on task {task_id}")
            else:
                total_preds = task.get('total_predictions', 0)
                # Some API versions return the actual list if expanded
                pred_list = task.get('predictions', []) 
                
                if total_preds > 0 or len(pred_list) > 0:
                    logger.debug(f"Task {task_id}: Already has predictions. Skipping.")
                    continue
            
            filename = image_uri.split('/')[-1]
            base_name = filename.rsplit('.', 1)[0]

            try:
                base_name, image_id = base_name.split('_')
                tax_rank, tax_name, tax_id = base_name.split('-')[:3]
            except Exception as e:
                logger.debug(f"Task {task_id}: Skipping taxonomy parse due to irregular filename - {e}")
                continue

            taxonomic_rank = tax_rank.capitalize()
            taxonomic_name = f'{tax_name.capitalize()} [{tax_id}]'

            uri_parts = image_uri.replace("s3://", "").split("/", 1)
            object_key = uri_parts[1]

            try:
                img_obj = s3_client.get_object(Bucket=args.bucket, Key=object_key)
                image = Image.open(BytesIO(img_obj['Body'].read())).convert("RGB")
            except Exception as e:
                logger.error(f"Task {task_id}: Failed to load image {object_key} from MinIO - {e}")
                continue
                
            width, height = image.size

            inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=args.threshold,
                text_threshold=args.threshold,
                target_sizes=[image.size[::-1]]
            )[0]

            prediction_results = []
            for box, score in zip(results["boxes"], results["scores"]):
                xmin, ymin, xmax, ymax = box.tolist()
                
                x_pct = (xmin / width) * 100
                y_pct = (ymin / height) * 100
                w_pct = ((xmax - xmin) / width) * 100
                h_pct = ((ymax - ymin) / height) * 100

                prediction_results.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": width,
                    "original_height": height,
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rotation": 0,
                        "rectanglelabels": [taxonomic_rank]
                    },
                    "score": float(score),
                    "project_id": args.project_id,
                })
                prediction_results.append({
                    "from_name": "rank",
                    "to_name": "image",
                    "type": "textarea",
                    "origin": "auto",
                    "value": {"text": [taxonomic_name]},
                    "score": 1.0,
                    "project_id": args.project_id,
                })

            if not prediction_results:
                logger.debug(f"Task {task_id}: No objects found matching prompt.")
                continue

            prediction_payload = {
                "project_id": args.project_id,
                "task": task_id,
                "model_version": "GroundingDINO-Tiny",
                "result": prediction_results
            }
            pred_url = f"{args.ls_url}/api/predictions/"
            post_resp = create_predictions(pred_url=pred_url, payload=prediction_payload, ls_headers=ls_headers, ls_url=args.ls_url, ls_refresh_token=ls_refresh_token)

            if post_resp.status_code == 201:
                total_processed += 1
                logger.info(f"Successfully added predictions to {total_processed} tasks...")
            else:
                logger.error(f"Task {task_id} failed pushing predictions: {post_resp.text}")

        page_no += 1
        tasks_url = f'{og_tasks_url}&page={page_no}'

if __name__ == "__main__":
    main()
