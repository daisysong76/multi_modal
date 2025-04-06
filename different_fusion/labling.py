import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import seaborn as sns
from datetime import datetime, timedelta

class AnnotationGuidelines:
    def __init__(self, n_annotators=5, n_samples=100):
        self.n_annotators = n_annotators
        self.n_samples = n_samples
        self.ground_truth = None
        self.annotations_before = None
        self.annotations_after = None
        
    def generate_sample_data(self):
        """Generate synthetic multimodal data samples (SDO images + radiation logs)"""
        print("Generating synthetic multimodal data samples...")
        
        # Simulate ground truth for solar events (0: No event, 1: C-class flare, 2: M-class flare, 3: X-class flare)
        self.ground_truth = np.random.choice([0, 1, 2, 3], size=self.n_samples, 
                                            p=[0.7, 0.15, 0.1, 0.05])  # Realistic class distribution
        
        # Create sample metadata
        dates = [datetime(2023, 1, 1) + timedelta(hours=i*6) for i in range(self.n_samples)]
        
        # Create DataFrame to represent the dataset
        self.dataset = pd.DataFrame({
            'sample_id': range(1, self.n_samples + 1),
            'timestamp': dates,
            'true_class': self.ground_truth,
            'class_name': [self._get_class_name(c) for c in self.ground_truth],
            'image_path': [f"sdo_image_{i+1}.fits" for i in range(self.n_samples)],
            'radiation_log': [f"radiation_log_{i+1}.csv" for i in range(self.n_samples)]
        })
        
        print(f"Generated {self.n_samples} samples with class distribution:")
        print(self.dataset['class_name'].value_counts())
        
    def _get_class_name(self, class_id):
        """Convert class ID to meaningful name"""
        classes = {0: "No Event", 1: "C-Class Flare", 2: "M-Class Flare", 3: "X-Class Flare"}
        return classes.get(class_id, "Unknown")
        
    def simulate_annotations_before_guidelines(self, error_rate=0.35):
        """Simulate annotations before improved guidelines with higher error rates"""
        print("\nSimulating annotations BEFORE guideline improvements...")
        
        # Initialize annotations matrix: n_annotators x n_samples
        self.annotations_before = np.zeros((self.n_annotators, self.n_samples), dtype=int)
        
        for annotator in range(self.n_annotators):
            for sample in range(self.n_samples):
                true_label = self.ground_truth[sample]
                
                # Simulate errors based on class (more complex events are harder to annotate correctly)
                if true_label == 0:  # No event - easier to identify
                    error_prob = error_rate * 0.5
                else:  # Flare events - harder to categorize correctly
                    error_prob = error_rate * (1 + true_label * 0.2)  # Higher classes have higher error rates
                
                if np.random.random() < error_prob:
                    # Make a mistake - either miss the event or misclassify its magnitude
                    if true_label == 0:
                        # False positive - label as a minor event
                        self.annotations_before[annotator, sample] = 1
                    else:
                        # Either miss the event or confuse its magnitude
                        possible_errors = [0]  # Miss the event
                        if true_label > 1:
                            possible_errors.append(true_label - 1)  # Underestimate
                        if true_label < 3:
                            possible_errors.append(true_label + 1)  # Overestimate
                        
                        self.annotations_before[annotator, sample] = np.random.choice(possible_errors)
                else:
                    # Correct annotation
                    self.annotations_before[annotator, sample] = true_label
        
        # Calculate inter-annotator agreement
        agreement = self._calculate_agreement(self.annotations_before)
        print(f"Average inter-annotator agreement before: {agreement:.2%}")
        
        # Calculate accuracy against ground truth
        accuracy = self._calculate_accuracy(self.annotations_before)
        print(f"Average annotator accuracy before: {accuracy:.2%}")
        
        return agreement, accuracy
    
    def simulate_annotations_after_guidelines(self, error_rate=0.1):
        """Simulate annotations after improved guidelines with lower error rates"""
        print("\nSimulating annotations AFTER guideline improvements...")
        
        # Initialize annotations matrix: n_annotators x n_samples
        self.annotations_after = np.zeros((self.n_annotators, self.n_samples), dtype=int)
        
        for annotator in range(self.n_annotators):
            for sample in range(self.n_samples):
                true_label = self.ground_truth[sample]
                
                # Simulate errors with reduced rates after better guidelines
                if true_label == 0:  # No event - easier to identify
                    error_prob = error_rate * 0.5
                else:  # Flare events - harder to categorize correctly
                    error_prob = error_rate * (1 + true_label * 0.1)  # Reduced influence of class difficulty
                
                if np.random.random() < error_prob:
                    # Make a mistake - but with better understanding of edge cases
                    if true_label == 0:
                        # False positive - label as a minor event
                        self.annotations_after[annotator, sample] = 1
                    else:
                        # Either miss the event or confuse its magnitude, but less likely to miss entirely
                        if true_label > 1:
                            possible_errors = [true_label - 1]  # More likely to just be off by one category
                            if np.random.random() < 0.2:  # Much less likely to miss entirely
                                possible_errors.append(0)
                        else:
                            possible_errors = [0, 2]  # For C-class, either miss or overestimate
                        
                        self.annotations_after[annotator, sample] = np.random.choice(possible_errors)
                else:
                    # Correct annotation
                    self.annotations_after[annotator, sample] = true_label
        
        # Calculate inter-annotator agreement
        agreement = self._calculate_agreement(self.annotations_after)
        print(f"Average inter-annotator agreement after: {agreement:.2%}")
        
        # Calculate accuracy against ground truth
        accuracy = self._calculate_accuracy(self.annotations_after)
        print(f"Average annotator accuracy after: {accuracy:.2%}")
        
        return agreement, accuracy
    
    def _calculate_agreement(self, annotations):
        """Calculate average pairwise Cohen's Kappa between annotators"""
        kappa_sum = 0
        comparisons = 0
        
        for i in range(self.n_annotators):
            for j in range(i+1, self.n_annotators):
                kappa = cohen_kappa_score(annotations[i], annotations[j])
                kappa_sum += kappa
                comparisons += 1
        
        return kappa_sum / comparisons if comparisons > 0 else 0
    
    def _calculate_accuracy(self, annotations):
        """Calculate average accuracy of annotators against ground truth"""
        accuracy_sum = 0
        
        for i in range(self.n_annotators):
            matches = (annotations[i] == self.ground_truth).sum()
            accuracy = matches / self.n_samples
            accuracy_sum += accuracy
        
        return accuracy_sum / self.n_annotators
    
    def analyze_confusion_patterns(self):
        """Analyze where annotators tend to make mistakes"""
        if self.annotations_before is None or self.annotations_after is None:
            print("Must run simulations first")
            return
            
        print("\nAnalyzing confusion patterns before guideline improvements:")
        all_annotations_before = self.annotations_before.flatten()
        all_ground_truth = np.tile(self.ground_truth, self.n_annotators)
        
        cm_before = confusion_matrix(all_ground_truth, all_annotations_before, 
                                    labels=[0, 1, 2, 3])
        
        print("\nConfusion matrix before guideline improvements:")
        print(cm_before)
        
        print("\nAnalyzing confusion patterns after guideline improvements:")
        all_annotations_after = self.annotations_after.flatten()
        
        cm_after = confusion_matrix(all_ground_truth, all_annotations_after,
                                   labels=[0, 1, 2, 3])
        
        print("\nConfusion matrix after guideline improvements:")
        print(cm_after)
        
        return cm_before, cm_after
    
    def visualize_improvements(self):
        """Create visualizations of the improvements"""
        if self.annotations_before is None or self.annotations_after is None:
            print("Must run simulations first")
            return
        
        # Calculate pairwise agreement scores before
        before_scores = []
        after_scores = []
        
        for i in range(self.n_annotators):
            for j in range(i+1, self.n_annotators):
                before_scores.append(cohen_kappa_score(self.annotations_before[i], 
                                                      self.annotations_before[j]))
                after_scores.append(cohen_kappa_score(self.annotations_after[i], 
                                                    self.annotations_after[j]))
        
        # Plot agreement improvements
        plt.figure(figsize=(10, 6))
        plt.hist([before_scores, after_scores], bins=10, 
                 label=['Before Guidelines', 'After Guidelines'],
                 alpha=0.7, color=['red', 'green'])
        plt.axvline(np.mean(before_scores), color='red', linestyle='dashed', linewidth=2)
        plt.axvline(np.mean(after_scores), color='green', linestyle='dashed', linewidth=2)
        plt.xlabel("Cohen's Kappa Score")
        plt.ylabel("Frequency")
        plt.title("Inter-annotator Agreement Before and After Guideline Improvements")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create confusion matrices visualizations
        cm_before, cm_after = self.analyze_confusion_patterns()
        
        class_names = ["No Event", "C-Class", "M-Class", "X-Class"]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(cm_before, annot=True, fmt='d', cmap='Reds', xticklabels=class_names, 
                    yticklabels=class_names)
        plt.title("Confusion Matrix Before Guideline Improvements")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        plt.subplot(1, 2, 2)
        sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, 
                   yticklabels=class_names)
        plt.title("Confusion Matrix After Guideline Improvements")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        plt.tight_layout()
        plt.show()
        
    def document_improvements(self):
        """Generate a report documenting the guideline improvements"""
        print("\n=== Annotation Guidelines Improvement Report ===")
        print("Project: NASA FDL Multimodal Solar Event Analysis")
        print("Data: Combined SDO images and radiation sensor logs")
        print(f"Dataset size: {self.n_samples} samples")
        print(f"Annotation team: {self.n_annotators} annotators")
        
        before_agreement, before_accuracy = self._calculate_agreement(self.annotations_before), self._calculate_accuracy(self.annotations_before)
        after_agreement, after_accuracy = self._calculate_agreement(self.annotations_after), self._calculate_accuracy(self.annotations_after)
        
        print("\nKey Metrics:")
        print(f"- Inter-annotator agreement before: {before_agreement:.2%}")
        print(f"- Inter-annotator agreement after: {after_agreement:.2%}")
        print(f"- Improvement: {after_agreement - before_agreement:.2%}")
        print(f"- Accuracy before: {before_accuracy:.2%}")
        print(f"- Accuracy after: {after_accuracy:.2%}")
        print(f"- Accuracy improvement: {after_accuracy - before_accuracy:.2%}")
        
        print("\nKey Guideline Improvements:")
        print("1. Added more visual examples of edge cases for each flare class")
        print("2. Created clearer distinction between C-class and M-class events")
        print("3. Improved instructions for correlating image data with radiation measurements")
        print("4. Added decision tree for ambiguous cases")
        print("5. Included more detailed explanation of annotation workflow")
        
        return {
            "before_agreement": before_agreement,
            "after_agreement": after_agreement,
            "agreement_improvement": after_agreement - before_agreement,
            "before_accuracy": before_accuracy,
            "after_accuracy": after_accuracy,
            "accuracy_improvement": after_accuracy - before_accuracy
        }


# Run a simulation of the annotation guidelines improvement
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create simulation with 8 annotators and 200 samples
    sim = AnnotationGuidelines(n_annotators=8, n_samples=200)
    
    # Generate synthetic data
    sim.generate_sample_data()
    
    # Simulate annotations before guideline improvements (35% error rate)
    sim.simulate_annotations_before_guidelines(error_rate=0.35)
    
    # Simulate annotations after guideline improvements (10% error rate)
    sim.simulate_annotations_after_guidelines(error_rate=0.10)
    
    # Analyze confusion patterns
    sim.analyze_confusion_patterns()
    
    # Generate report
    metrics = sim.document_improvements()
    
    print("\nSimulation complete. This script demonstrates how annotation guidelines")
    print("improvements led to substantial increases in inter-annotator agreement")
    print(f"from {metrics['before_agreement']:.2%} to {metrics['after_agreement']:.2%}.")