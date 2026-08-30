#!/usr/bin/env python3
# /// script
# dependencies = [
#   "boto3",
#   "datasets",
#   "Pillow",
# ]
# ///

import argparse
import os
import sys
import boto3
from datasets import load_dataset
from io import BytesIO
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Stream images from Hugging Face directly into a MinIO bucket.")
    parser.add_argument("dataset", help="Hugging Face dataset name (e.g., 'alan-turing-institute/insects')")
    parser.add_argument("--split", default="train", help="Dataset split to load (default: train)")
    parser.add_argument("--image-col", default="image", help="Column name containing the PIL image (default: image)")
    parser.add_argument("--bucket", required=True, help="Target MinIO bucket name")
    parser.add_argument("--prefix", default="hf_import", help="Folder prefix in the bucket (default: hf_import)")
    parser.add_argument("--endpoint", default="https://api.s3.local.godesteem.de", help="MinIO API endpoint URL")
    parser.add_argument("--limit", type=int, help="Maximum number of images to ingest (optional)")

    args = parser.parse_args()

    # Pull secrets from environment
    access_key = os.environ.get("MINIO_ACCESS_KEY")
    secret_key = os.environ.get("MINIO_SECRET_KEY")

    if not access_key or not secret_key:
        print("Error: MINIO_ACCESS_KEY and MINIO_SECRET_KEY environment variables must be set.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to MinIO at {args.endpoint}...")
    s3_client = boto3.client(
        's3',
        endpoint_url=args.endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='us-east-1'
    )

    print(f"Loading dataset '{args.dataset}' (split: {args.split})...")
    # streaming=True ensures we don't download the whole dataset to RAM
    dataset = load_dataset(args.dataset, split=args.split, streaming=True)

    print(f"Streaming images into bucket '{args.bucket}/{args.prefix}'...")
    
    count = 0
    for row in dataset:
        if args.limit and count >= args.limit:
            break
            
        try:
            image_data = row[args.image_col]
            
            # 1. Handle datasets that return raw bytes or dicts with a 'bytes' key
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(BytesIO(image_data['bytes']))
            elif isinstance(image_data, bytes):
                image = Image.open(BytesIO(image_data))
            else:
                # Assume it's already a PIL Image
                image = image_data
            
            # 2. Convert to RGB if it's grayscale or RGBA
            if image.mode != 'RGB':
                image = image.convert('RGB')

            name = f"{row['taxonomic_rank']}-{row['species_name']}-{row['taxon_id']}"
                
            # 3. Save to byte stream for MinIO
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            object_key = f"{args.prefix}/{name}_{count:06d}.jpg"
            
            s3_client.upload_fileobj(
                img_byte_arr, 
                args.bucket, 
                object_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            count += 1
            if count % 100 == 0:
                print(f"Successfully uploaded {count} images...")
                
        except Exception as e:
            print(f"Warning: Failed to process image {count}. Error: {e}", file=sys.stderr)
            # Add this to skip the failing image but increment count so you don't get stuck
            count += 1

    print(f"Ingestion complete. {count} total images uploaded to s3://{args.bucket}/{args.prefix}/")

if __name__ == "__main__":
    main()
