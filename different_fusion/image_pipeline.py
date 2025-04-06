
# import cv2
# import pytesseract
# from PIL import Image
# import torch
# from torchvision import models, transforms
# import json

# def detect_text(image_path):
#     image = Image.open(image_path)
#     text = pytesseract.image_to_string(image)
#     return text

# def classify_scene(image_path):
#     model = models.resnet18(pretrained=True)
#     model.eval()
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor()
#     ])
#     image = Image.open(image_path)
#     input_tensor = preprocess(image).unsqueeze(0)
#     with torch.no_grad():
#         output = model(input_tensor)
#     predicted_class = output.argmax().item()
#     return int(predicted_class)

# def image_pipeline(image_path, output_path="image_output.json"):
#     text = detect_text(image_path)
#     scene_class = classify_scene(image_path)
#     output = {
#         "detected_text": text,
#         "scene_class_id": scene_class
#     }
#     with open(output_path, "w") as f:
#         json.dump(output, f)

# Example usage
# image_pipeline("example.jpg")

import boto3
import json
import os
import logging
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process images using Amazon Bedrock for text detection, logo detection, content moderation, and scene classification."""
    
    def __init__(self, region_name='us-east-1'):
        """Initialize the image processor with AWS credentials."""
        self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        self.s3 = boto3.client('s3', region_name=region_name)
        self.rekognition = boto3.client('rekognition', region_name=region_name)
        self.output_bucket = os.environ.get('OUTPUT_BUCKET', 'image-processing-output')
        
    def process_image(self, image_path, tasks=None):
        """
        Process an image using Amazon Bedrock and Rekognition.
        
        Args:
            image_path: Path to the image file
            tasks: List of tasks to perform. If None, performs all tasks.
                  Available tasks: 'text_detection', 'logo_detection', 'moderation', 'scene_classification'
        
        Returns:
            Dictionary with processing results
        """
        if tasks is None:
            tasks = ['text_detection', 'logo_detection', 'moderation', 'scene_classification']
        
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
                
            # Convert to base64 for Bedrock models
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Process with different capabilities
            results = {"image_name": os.path.basename(image_path)}
            
            if 'text_detection' in tasks:
                results['text_detection'] = self._detect_text(image_bytes)
            
            if 'logo_detection' in tasks:
                results['logo_detection'] = self._detect_logos(image_bytes)
            
            if 'moderation' in tasks:
                results['moderation'] = self._moderate_content(image_bytes)
            
            if 'scene_classification' in tasks:
                results['scene_classification'] = self._classify_scene(image_bytes, image_base64)
            
            # Save annotated image if text or logos were detected
            if 'text_detection' in tasks or 'logo_detection' in tasks:
                annotated_image = self._create_annotated_image(
                    image_path, 
                    results.get('text_detection', {}).get('text_detections', []),
                    results.get('logo_detection', {}).get('logos', [])
                )
                
                # Save annotated image to S3
                filename = os.path.basename(image_path)
                s3_key = f"annotated/{datetime.now().strftime('%Y%m%d%H%M%S')}/{filename}"
                
                buffer = BytesIO()
                annotated_image.save(buffer, format="JPEG")
                buffer.seek(0)
                
                self.s3.upload_fileobj(buffer, self.output_bucket, s3_key)
                results['annotated_image_url'] = f"s3://{self.output_bucket}/{s3_key}"
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return {"error": str(e)}
    
    def _detect_text(self, image_bytes):
        """Detect text in an image using Amazon Rekognition."""
        response = self.rekognition.detect_text(Image={'Bytes': image_bytes})
        
        # Extract and structure text detections
        text_detections = []
        detected_text = ""
        
        for detection in response.get('TextDetections', []):
            if detection['Type'] == 'WORD':
                text_detections.append({
                    'text': detection['DetectedText'],
                    'confidence': detection['Confidence'],
                    'bounding_box': detection['Geometry']['BoundingBox']
                })
                detected_text += detection['DetectedText'] + " "
        
        return {
            "text_detections": text_detections,
            "full_text": detected_text.strip()
        }
    
    def _detect_logos(self, image_bytes):
        """Detect logos in an image using Amazon Rekognition."""
        response = self.rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            Features=['GENERAL_LABELS']
        )
        
        # Extract logos from labels
        logos = []
        for label in response.get('Labels', []):
            if label.get('Categories', []):
                for category in label['Categories']:
                    if category['Name'] == 'Brand' or category['Name'] == 'Logo':
                        logos.append({
                            'name': label['Name'],
                            'confidence': label['Confidence'],
                            'bounding_box': self._get_bounding_box(label)
                        })
        
        return {"logos": logos}
    
    def _get_bounding_box(self, label):
        """Extract bounding box from a label if available."""
        if 'Instances' in label and label['Instances']:
            return label['Instances'][0]['BoundingBox']
        return None
    
    def _moderate_content(self, image_bytes):
        """Perform content moderation on an image using Amazon Rekognition."""
        response = self.rekognition.detect_moderation_labels(
            Image={'Bytes': image_bytes},
            MinConfidence=50
        )
        
        # Structure moderation results
        moderation_labels = []
        for label in response.get('ModerationLabels', []):
            moderation_labels.append({
                'name': label['Name'],
                'parent_name': label.get('ParentName', ''),
                'confidence': label['Confidence']
            })
        
        moderation_result = {
            "moderation_labels": moderation_labels,
            "flagged_content": len(moderation_labels) > 0
        }
        
        return moderation_result
    
    def _classify_scene(self, image_bytes, image_base64):
        """Classify scene in an image using Amazon Bedrock foundation model."""
        # Use Claude to classify the scene
        prompt = f"""
        I'll show you an image. Please classify the scene in the image by describing:
        1. The overall scene type (indoor, outdoor, urban, rural, etc.)
        2. Key objects visible in the scene
        3. The general atmosphere or context
        4. Any notable activities happening
        
        Format your response as JSON with these fields:
        {{
            "scene_type": "string",
            "key_objects": ["string", "string", ...],
            "atmosphere": "string",
            "activities": ["string", "string", ...]
        }}
        """
        
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        
        response = self.bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ]
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        scene_analysis = response_body['content'][0]['text']
        
        # Try to extract JSON from the response
        try:
            # Find JSON in text response
            json_start = scene_analysis.find('{')
            json_end = scene_analysis.rfind('}')
            
            if json_start >= 0 and json_end >= 0:
                json_str = scene_analysis[json_start:json_end+1]
                return json.loads(json_str)
            else:
                return {"description": scene_analysis}
        except json.JSONDecodeError:
            return {"description": scene_analysis}
    
    def _create_annotated_image(self, image_path, text_detections, logos):
        """Create annotated image showing detected text and logos."""
        # Open image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw text detections
        for detection in text_detections:
            if 'bounding_box' in detection:
                box = detection['bounding_box']
                width, height = image.size
                
                left = box['Left'] * width
                top = box['Top'] * height
                right = left + (box['Width'] * width)
                bottom = top + (box['Height'] * height)
                
                draw.rectangle([left, top, right, bottom], outline="blue", width=2)
                draw.text((left, top-15), detection['text'], fill="blue", font=font)
        
        # Draw logo detections
        for logo in logos:
            if logo.get('bounding_box'):
                box = logo['bounding_box']
                width, height = image.size
                
                left = box['Left'] * width
                top = box['Top'] * height
                right = left + (box['Width'] * width)
                bottom = top + (box['Height'] * height)
                
                draw.rectangle([left, top, right, bottom], outline="green", width=2)
                draw.text((left, bottom+5), f"Logo: {logo['name']}", fill="green", font=font)
        
        return image
    
    def batch_process(self, directory_path, tasks=None):
        """Process all images in a directory."""
        results = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                file_path = os.path.join(directory_path, filename)
                logger.info(f"Processing {file_path}")
                results[filename] = self.process_image(file_path, tasks)
        return results

if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor()
    
    # Process a single image with all tasks
    result = processor.process_image("sample_image.jpg")
    print(json.dumps(result, indent=2))
    
    # Process all images in a directory with specific tasks
    # batch_results = processor.batch_process("./images", tasks=['text_detection', 'scene_classification'])
