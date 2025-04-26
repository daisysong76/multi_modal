it need to be runned through docker:
/Users/songxiaomei/Desktop/multi_modal

cd simple_docker && ./build.sh && docker run -it --memory=8g --memory-swap=16g --shm-size=1g -v /Users/songxiaomei/Desktop/multi_modal:/app/input semantic-alignment

cd simple_docker && chmod +x build.sh && ./build.sh && docker run -it --memory=8g --memory-swap=16g --shm-size=1g -v /Users/songxiaomei/Desktop/multi_modal:/app/input semantic-alignment



The code ran successfully and completed the semantic alignment analysis. Here's what happened:
1. Video Processing:
Successfully loaded and processed the video
Extracted video features with shape: [1, 2, 1024] (1 batch, 2 frames, 1024 features)
2. Audio Processing:
Successfully extracted audio from video
Trimmed audio from 17,188,800 to 960,000 samples (60 seconds)
Extracted audio features with shape: [1, 373, 512] (1 batch, 373 time steps, 512 features)
3. Cross-Modal Analysis:
Successfully performed cross-modal attention
Generated predicted audio features: [1, 2, 512]
Computed alignment scores: [1, 375, 1]
4. Memory Usage:
Peak memory usage was around 2.5GB RAM
Successfully cleaned up memory after processing
Final memory usage: 1.2GB RAM

The model successfully analyzed the semantic alignment between the video and audio. Would you like me to:
1. Save the alignment scores to a file for analysis?
2. Visualize the alignment scores?
3. Process a different video?
4. Make any adjustments to the model parameters?
