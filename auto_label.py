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
import argparse
import requests
import boto3
import torch
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def main():
    parser = argparse.ArgumentParser(description="Auto-label Label Studio tasks using Grounding DINO.")
    parser.add_argument("--ls-url", default="https://label-studio.local.godesteem.de", help="Label Studio URL")
    parser.add_argument("--project-id", required=True, type=int, help="Label Studio Project ID")
    parser.add_argument("--prompt", default="insect .", help="Text prompt for Grounding DINO (e.g., 'insect . bug . bee .')")
    parser.add_argument("--threshold", default=0.3, type=float, help="Confidence threshold (0.0 to 1.0)")
    parser.add_argument("--bucket", default="insects", help="MinIO bucket name")
    parser.add_argument("--minio-endpoint", default="https://api.s3.local.godesteem.de", help="Internal MinIO/Traefik IP")
    parser.add_argument("--clear-existing", action="store_true", help="Delete existing predictions on the task before adding new ones")    
    args = parser.parse_args()

    # Pull credentials from environment
    ls_refresh_token = os.environ.get("LABEL_STUDIO_TOKEN")
    minio_ak = os.environ.get("MINIO_ACCESS_KEY")
    minio_sk = os.environ.get("MINIO_SECRET_KEY")

    if not all([ls_refresh_token, minio_ak, minio_sk]):
        print("Missing required environment variables: LABEL_STUDIO_TOKEN, MINIO_ACCESS_KEY, MINIO_SECRET_KEY")
        exit(1)

    print("Loading Grounding DINO (this may take a moment)...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    print(f"Connecting to MinIO...")
    s3_client = boto3.client(
        's3',
        endpoint_url=args.minio_endpoint,
        aws_access_key_id=minio_ak,
        aws_secret_access_key=minio_sk,
        region_name='us-east-1'
    )

    # get ls Token
    token_url = f'{args.ls_url}/api/token/refresh'
    payload = {'refresh': ls_refresh_token}
    response = requests.post(token_url, json=payload)
    tokens = response.json()
    ls_token = tokens['access']

    ls_headers = {"Authorization": f"Bearer {ls_token}"}
    
    # 1. Fetch all tasks from the Label Studio project
    print(f"Fetching tasks for Project {args.project_id}...")
    
    # We set page_size to 100 to reduce the number of API calls
    og_tasks_url = f"{args.ls_url}/api/projects/{args.project_id}/tasks?page_size=100"
    
    page_no = 1
    tasks_url = f'{og_tasks_url}&page={page_no}'
    total_processed = 0

    while tasks_url:
        print(f"Fetching page: {tasks_url}")
        response = requests.get(tasks_url, headers=ls_headers)
        
        if response.status_code != 200:
            print(f"Error fetching tasks: {response.text}")
            break
            
        response_data = response.json()
        
        # Handle DRF paginated dicts vs legacy flat lists
        if isinstance(response_data, dict):
            tasks = response_data.get('results', response_data.get('tasks', []))
            # DRF provides the exact URL for the next page
            tasks_url = response_data.get('next') 
        else:
            # Fallback if the API ever returns an unpaginated raw list
            tasks = response_data
            tasks_url = None
            
        if not tasks:
            break

        # Process the batch of tasks on the current page
        for task in tasks:
            task_id = task['id']
            
            image_uri = task['data'].get('image', '')
            if not image_uri.startswith('s3://'):
                continue
            
            # --- CLEAN EXISTING PREDICTIONS ---
            if args.clear_existing:
                # Fetch all current predictions for this specific task
                pred_url = f"{args.ls_url}/api/predictions/?task={task_id}"
                pred_resp = requests.get(pred_url, headers=ls_headers)
                
                if pred_resp.status_code == 200:
                    for pred in pred_resp.json():
                        pred_id = pred.get('id')
                        if pred_id:
                            del_resp = requests.delete(f"{args.ls_url}/api/predictions/{pred_id}/", headers=ls_headers)
                            if del_resp.status_code not in (200, 204):
                                print(f"Warning: Failed to delete prediction {pred_id} on task {task_id}")
            # ----------------------------------    
            # --- FILENAME EXTRACTION ---
            # 1. Get the raw filename (e.g., "Apis_mellifera_001.jpg")
            filename = image_uri.split('/')[-1]
            
            # 2. Strip the extension
            base_name = filename.rsplit('.', 1)[0]

            # remove image ID
            base_name, image_id = base_name.split('_')
            # reconstruct identifier
            tax_rank, tax_name, tax_id = base_name.split('-')
            
            taxonomic_rank = tax_rank.capitalize()
            taxonomic_name = f'{tax_name.capitalize()} [{tax_id}]'

            # Parse bucket and object key for downloading
            uri_parts = image_uri.replace("s3://", "").split("/", 1)
            object_key = uri_parts[1]

            # 2. Download image into memory directly from MinIO
            try:
                img_obj = s3_client.get_object(Bucket=args.bucket, Key=object_key)
                image = Image.open(BytesIO(img_obj['Body'].read())).convert("RGB")
            except Exception as e:
                print(f"Task {task_id}: Failed to load image {object_key} from MinIO: {e}")
                continue
                
            width, height = image.size

            # 3. Run Grounding DINO Inference
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

            # 4. Format predictions for Label Studio
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
                        # --- INJECT THE EXTRACTED RANK HERE ---
                        "rectanglelabels": [taxonomic_name]
                    },
                    "score": float(score)
                })
                prediction_results.append({
                    "from_name": "rank",
                    "to_name": "image",
                    "type": "textarea",
                    "origin": "auto",
                    "value": {"text": [taxonomic_rank]},
                    "score": 1.0
                })

            if not prediction_results:
                print(f"Task {task_id}: No objects found.")
                continue

            # 5. Push the prediction to Label Studio
            prediction_payload = {
                "task": task_id,
                "model_version": "GroundingDINO-Tiny",
                "result": prediction_results
            }
            
            post_resp = requests.post(
                f"{args.ls_url}/api/predictions/", 
                headers=ls_headers, 
                json=prediction_payload
            )
            
            if post_resp.status_code == 201:
                total_processed += 1
                if total_processed % 10 == 0:
                    print(f"Successfully added predictions to {total_processed} tasks...")
            else:
                print(f"Task {task_id} failed: {post_resp.text}")

        page_no += 1
        tasks_url = f'{og_tasks_url}&page={page_no}'

        if not prediction_results:
            print(f"Task {task_id}: No objects found.")
            continue

        # 5. Push the prediction to Label Studio
        prediction_payload = {
            "task": task_id,
            "model_version": "GroundingDINO-Tiny",
            "result": prediction_results
        }
        
        post_resp = requests.post(
            f"{args.ls_url}/api/predictions/", 
            headers=ls_headers, 
            json=prediction_payload
        )
        
        if post_resp.status_code == 201:
            print(f"Task {task_id}: Added {len(prediction_results)} predictions.")
        else:
            print(f"Task {task_id} failed: {post_resp.text}")


if __name__ == "__main__":
    main()
