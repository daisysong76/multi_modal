
# import fitz  # PyMuPDF
# import re
# import json
# from transformers import AutoTokenizer, AutoModel
# import torch

# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# def get_semantic_representation(text):
#     tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#     model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         embeddings = model(**inputs).last_hidden_state.mean(dim=1)
#     return embeddings.numpy().tolist()

# def document_pipeline(pdf_path, output_path="semantic_output.json"):
#     raw_text = extract_text_from_pdf(pdf_path)
#     embeddings = get_semantic_representation(raw_text)
#     with open(output_path, "w") as f:
#         json.dump({"text": raw_text, "embedding": embeddings}, f)

# Example usage
# document_pipeline("example.pdf")


import boto3
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents using Amazon Bedrock to extract structured data from unstructured content."""
    
    def __init__(self, region_name='us-east-1'):
        """Initialize the document processor with AWS credentials."""
        self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        self.bedrock_agent = boto3.client('bedrock-agent', region_name=region_name)
        self.s3 = boto3.client('s3', region_name=region_name)
        self.output_bucket = os.environ.get('OUTPUT_BUCKET', 'document-processing-output')
        
    def process_document(self, file_path, model_id="anthropic.claude-3-sonnet-20240229-v1:0"):
        """Process a document using Amazon Bedrock."""
        try:
            # Determine file type and read content
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension in ['.pdf', '.docx', '.doc']:
                # For complex documents, use Document processing API
                return self._process_complex_document(file_path)
            else:
                # For text documents, use foundation model directly
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                return self._extract_semantic_content(content, model_id)
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {"error": str(e)}
    
    def _process_complex_document(self, file_path):
        """Process complex documents like PDFs with layout analysis."""
        # Upload document to S3 if not already there
        filename = os.path.basename(file_path)
        s3_key = f"input/{datetime.now().strftime('%Y%m%d%H%M%S')}/{filename}"
        
        with open(file_path, 'rb') as document:
            self.s3.upload_fileobj(document, self.output_bucket, s3_key)
        
        # Create a document processing job
        response = self.bedrock_agent.create_data_source(
            name=f"doc-processing-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            dataSourceConfiguration={
                's3Configuration': {
                    'bucketName': self.output_bucket,
                    'inclusionPrefixes': [f"input/{s3_key}"]
                }
            },
            documentProcessingConfiguration={
                'extractLayoutElements': True,
                'extractTitle': True,
                'extractTables': True,
                'documentSplitterConfiguration': {
                    'chunkingStrategy': 'SEMANTIC',
                    'maxTokens': 1000,
                    'overlapPercentage': 20
                }
            }
        )
        
        job_id = response['dataSourceId']
        logger.info(f"Started document processing job: {job_id}")
        
        # Wait for job completion
        waiter = self.bedrock_agent.get_waiter('data_source_complete')
        waiter.wait(dataSourceId=job_id)
        
        # Get results
        results = self.bedrock_agent.get_data_source(dataSourceId=job_id)
        
        # Extract structured data from the processed document
        structured_data = {
            "document_id": job_id,
            "title": results.get('title', 'Unknown'),
            "sections": self._extract_sections(results),
            "tables": self._extract_tables(results),
            "metadata": results.get('metadata', {})
        }
        
        return structured_data
    
    def _extract_sections(self, processed_document):
        """Extract sections from processed document."""
        # Implementation would parse the processed document to identify sections
        sections = []
        # In a real implementation, we would extract sections from the processed document
        return sections
    
    def _extract_tables(self, processed_document):
        """Extract tables from processed document."""
        # Implementation would parse the processed document to identify tables
        tables = []
        # In a real implementation, we would extract tables from the processed document
        return tables
    
    def _extract_semantic_content(self, text, model_id):
        """Extract semantic representations from text using foundation model."""
        prompt = f"""
        Please analyze the following document text and extract key information:
        1. Main topics and themes
        2. Key entities (people, organizations, locations)
        3. Important facts and figures
        4. Relationships between entities
        5. Document structure summary
        
        Format the output as structured JSON.
        
        Document Text:
        {text}
        """
        
        response = self.bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        response_body = json.loads(response['body'].read().decode('utf-8'))
        structured_content = response_body['content'][0]['text']
        
        # Convert the response to a Python dictionary
        try:
            # Extract JSON from the response which might contain markdown
            json_start = structured_content.find('{')
            json_end = structured_content.rfind('}')
            
            if json_start >= 0 and json_end >= 0:
                json_str = structured_content[json_start:json_end+1]
                return json.loads(json_str)
            else:
                return {"text": structured_content}
        except json.JSONDecodeError:
            return {"text": structured_content}
    
    def batch_process(self, directory_path):
        """Process all documents in a directory."""
        results = {}
        for filename in os.listdir(directory_path):
            if filename.endswith(('.pdf', '.docx', '.doc', '.txt')):
                file_path = os.path.join(directory_path, filename)
                logger.info(f"Processing {file_path}")
                results[filename] = self.process_document(file_path)
        return results

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    
    # Process a single document
    result = processor.process_document("sample_document.pdf")
    print(json.dumps(result, indent=2))
    
    # Process all documents in a directory
    # batch_results = processor.batch_process("./documents")