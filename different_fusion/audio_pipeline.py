
# import whisper
# import torchaudio
# import json
# import os

# def transcribe_audio(audio_path):
#     model = whisper.load_model("base")
#     result = model.transcribe(audio_path)
#     return result["text"], result["segments"]

# def audio_pipeline(audio_path, output_path="audio_output.json"):
#     text, segments = transcribe_audio(audio_path)
#     output = {
#         "transcript": text,
#         "segments": segments
#     }
#     with open(output_path, "w") as f:
#         json.dump(output, f)

# Example usage
# audio_pipeline("example.mp3")


import boto3
import json
import os
import logging
import time
from datetime import datetime
from pydub import AudioSegment
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Process audio files using Amazon Bedrock for transcription, segmentation, and moderation."""
    
    def __init__(self, region_name='us-east-1'):
        """Initialize the audio processor with AWS credentials."""
        self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
        self.transcribe = boto3.client('transcribe', region_name=region_name)
        self.s3 = boto3.client('s3', region_name=region_name)
        self.comprehend = boto3.client('comprehend', region_name=region_name)
        self.output_bucket = os.environ.get('OUTPUT_BUCKET', 'audio-processing-output')
    
    def process_audio(self, audio_path, language_code='en-US', tasks=None):
        """
        Process an audio file using Amazon Bedrock and related services.
        
        Args:
            audio_path: Path to the audio file
            language_code: Language code for transcription
            tasks: List of tasks to perform. If None, performs all tasks.
                  Available tasks: 'transcription', 'segmentation', 'moderation'
        
        Returns:
            Dictionary with processing results
        """
        if tasks is None:
            tasks = ['transcription', 'segmentation', 'moderation']
        
        try:
            # Upload audio to S3 for processing
            filename = os.path.basename(audio_path)
            s3_key = f"input/{datetime.now().strftime('%Y%m%d%H%M%S')}/{filename}"
            
            with open(audio_path, 'rb') as audio_file:
                self.s3.upload_fileobj(audio_file, self.output_bucket, s3_key)
            
            s3_uri = f"s3://{self.output_bucket}/{s3_key}"
            
            results = {"audio_name": filename, "s3_uri": s3_uri}
            
            # Perform transcription first as it's needed for other tasks
            if 'transcription' in tasks or 'moderation' in tasks:
                transcription = self._transcribe_audio(audio_path, s3_uri, language_code)
                results['transcription'] = transcription
            
            if 'segmentation' in tasks:
                segments = self._segment_audio(audio_path, 
                                              results.get('transcription', {}).get('transcript', ''))
                results['segmentation'] = segments
            
            if 'moderation' in tasks and 'transcription' in results:
                moderation = self._moderate_content(results['transcription'].get('transcript', ''))
                results['moderation'] = moderation
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {str(e)}")
            return {"error": str(e)}
    
    def _transcribe_audio(self, audio_path, s3_uri, language_code='en-US'):
        """Transcribe audio using Amazon Transcribe and enhance with Amazon Bedrock."""
        file_name = os.path.basename(audio_path)
        job_name = f"transcription-{int(time.time())}"
        
        # Start transcription job
        response = self.transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat=os.path.splitext(file_name)[1][1:],  # Remove the dot from extension
            LanguageCode=language_code,
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 10
            }
        )
        
        # Wait for completion
        while True:
            status = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
        
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'FAILED':
            return {"error": "Transcription job failed"}
        
        # Get the transcript
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        
        # Download and parse transcript
        import urllib.request
        with urllib.request.urlopen(transcript_uri) as response:
            transcript_json = json.loads(response.read().decode('utf-8'))
        
        # Extract basic transcript and speaker labels
        full_transcript = transcript_json['results']['transcripts'][0]['transcript']
        
        # Extract time-aligned words
        items = transcript_json['results'].get('items', [])
        
        # Extract speaker segments if available
        speaker_segments = []
        if 'speaker_labels' in transcript_json['results']:
            speakers = transcript_json['results']['speaker_labels']
            segments = speakers.get('segments', [])
            
            for segment in segments:
                speaker_label = segment['speaker_label']
                start_time = float(segment['start_time'])
                end_time = float(segment['end_time'])
                
                # Collect all words spoken by this speaker in this segment
                segment_words = []
                for item in items:
                    if item['type'] == 'pronunciation':
                        if 'start_time' in item and 'end_time' in item:
                            word_start = float(item['start_time'])
                            if start_time <= word_start <= end_time:
                                segment_words.append(item['alternatives'][0]['content'])
                
                speaker_segments.append({
                    'speaker': speaker_label,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': ' '.join(segment_words)
                })
        
        # Enhance transcript with Bedrock if there's content
        enhanced_transcript = None
        if full_transcript:
            enhanced_transcript = self._enhance_transcript(full_transcript)
        
        return {
            "transcript": full_transcript,
            "speaker_segments": speaker_segments,
            "enhanced_transcript": enhanced_transcript
        }
    
    def _enhance_transcript(self, transcript):
        """Enhance transcript using Amazon Bedrock foundation model."""
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        
        prompt = f"""
        Please clean up and enhance the following transcript:
        1. Fix grammar and punctuation
        2. Remove filler words and hesitations
        3. Format speaker turns clearly
        4. Keep the original meaning intact
        
        Transcript:
        {transcript}
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
        enhanced_transcript = response_body['content'][0]['text']
        
        return enhanced_transcript
    
    def _segment_audio(self, audio_path, transcript=None):
        """Segment audio file based on silence detection and speaker changes."""
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Parameters for segmentation
            min_silence_len = 1000  # 1 second
            silence_thresh = -40  # dB
            
            # Detect silence
            from pydub.silence import detect_nonsilent
            non_silent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, 
                                               silence_thresh=silence_thresh)
            
            segments = []
            
            # Create segments based on non-silent ranges
            for i, (start_ms, end_ms) in enumerate(non_silent_ranges):
                segment_audio = audio[start_ms:end_ms]
                
                # Calculate audio features
                segment_duration = (end_ms - start_ms) / 1000  # in seconds
                
                # Convert to numpy array for feature extraction
                samples = np.array(segment_audio.get_array_of_samples())
                channels = segment_audio.channels
                
                # If stereo, take mean of channels
                if channels == 2:
                    samples = samples.reshape((-1, 2))
                    samples = samples.mean(axis=1)
                
                # Calculate audio features
                rms = np.sqrt(np.mean(samples**2))
                peak = np.max(np.abs(samples))
                
                segment_info = {
                    "id": i,
                    "start_time": start_ms / 1000,  # Convert to seconds
                    "end_time": end_ms / 1000,      # Convert to seconds
                    "duration": segment_duration,
                    "features": {
                        "rms": float(rms),
                        "peak": float(peak)
                    }
                }
                
                segments.append(segment_info)
                
                # Export segment to file
                segment_filename = f"segment_{i}_{start_ms}_{end_ms}.wav"
                segment_path = os.path.join("/tmp", segment_filename)
                segment_audio.export(segment_path, format="wav")
                
                # Upload segment to S3
                s3_segment_key = f"segments/{os.path.basename(audio_path)}/{segment_filename}"
                with open(segment_path, 'rb') as segment_file:
                    self.s3.upload_fileobj(segment_file, self.output_bucket, s3_segment_key)
                
                segment_info["s3_uri"] = f"s3://{self.output_bucket}/{s3_segment_key}"
                
                # Clean up temp file
                os.remove(segment_path)
            
            # If transcript is available, try to match segments with transcript
            if transcript:
                segments = self._match_transcript_to_segments(segments, transcript)
            
            return {
                "total_segments": len(segments),
                "total_duration": audio.duration_seconds,
                "segments": segments
            }
            
        except Exception as e:
            logger.error(f"Error in audio segmentation: {str(e)}")
            return {"error": str(e)}
    
    def _match_transcript_to_segments(self, segments, transcript):
        """Match transcript parts to audio segments using timing if available."""
        # This is a simplified implementation
        # In a real-world scenario, you would use the word timings from the transcript
        
        # If we have a short transcript and few segments, just distribute evenly
        if len(segments) <= 3:
            words = transcript.split()
            words_per_segment = len(words) // len(segments)
            
            for i, segment in enumerate(segments):
                start_idx = i * words_per_segment
                end_idx = (i + 1) * words_per_segment if i < len(segments) - 1 else len(words)
                segment_text = ' '.join(words[start_idx:end_idx])
                segments[i]["transcript"] = segment_text
        
        return segments
    
    def _moderate_content(self, transcript):
        """Perform content moderation on transcript using Amazon Comprehend."""
        if not transcript:
            return {"moderation_result": "No transcript to moderate"}
        
        # Check for toxicity/sensitive content using Comprehend
        try:
            response = self.comprehend.detect_toxic_content(
                TextSegments=[{"Text": transcript}],
                LanguageCode='en'
            )
            
            # Extract toxicity results
            toxic_labels = []
            for result in response.get('ResultList', []):
                for label in result.get('Labels', []):
                    if label['Score'] > 0.5:  # Only include high confidence toxic labels
                        toxic_labels.append({
                            'name': label['Name'],
                            'score': label['Score']
                        })
            
            return {
                "moderation_labels": toxic_labels,
                "flagged_content": len(toxic_labels) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in content moderation: {str(e)}")
            return {"error": str(e)}
    
    def batch_process(self, directory_path, language_code='en-US', tasks=None):
        """Process all audio files in a directory."""
        results = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                file_path = os.path.join(directory_path, filename)
                logger.info(f"Processing {file_path}")
                results[filename] = self.process_audio(file_path, language_code, tasks)
        return results

if __name__ == "__main__":
    # Example usage
    processor = AudioProcessor()
    
    # Process a single audio file with all tasks
    result = processor.process_audio("sample_audio.mp3")
    #print(json.dumps(result, indent=
