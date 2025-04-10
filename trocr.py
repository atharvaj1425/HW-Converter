# import cv2
# import numpy as np
# from PIL import Image
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# def preprocess_image(image_path, target_size=(384, 384)):
#     """
#     Load the image from `image_path`, resize it to `target_size`,
#     apply bilateral filtering for denoising and CLAHE for contrast enhancement.
#     Converts the grayscale image to RGB before returning.
#     """
#     # Load image in grayscale mode
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if image is None:
#         raise ValueError(f"Image not found at: {image_path}")
#     # Resize image
#     image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
#     # Apply bilateral filter to reduce noise while preserving edges
#     image_denoised = cv2.bilateralFilter(image_resized, d=9, sigmaColor=75, sigmaSpace=75)
#     # Apply CLAHE for contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     image_clahe = clahe.apply(image_denoised)
#     # Convert the grayscale image (2D) to an RGB image (3D)
#     image_rgb = cv2.cvtColor(image_clahe, cv2.COLOR_GRAY2RGB)
#     return Image.fromarray(image_rgb)

# # Load the TrOCR processor and model (using the large, handwritten variant)
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# def extract_text(image_path):
#     """
#     Preprocess the image and extract text using TrOCR.
#     """
#     image = preprocess_image(image_path)
#     pixel_values = processor(image, return_tensors="pt").pixel_values
#     generated_ids = model.generate(pixel_values)
#     text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     return text

# if __name__ == "__main__":
#     # Update with the correct path to your handwritten line image.
#     image_path = "Screenshot 2025-03-07 215422.png"
#     extracted_text = extract_text(image_path)
#     print("Extracted Text:", extracted_text)


import cv2
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from PIL import Image

class HandwritingExtractor:
    def __init__(self, trocr_model_name="microsoft/trocr-large-handwritten"):
        """
        Initialize the handwriting extractor with the TrOCR model.
        
        Args:
            trocr_model_name (str): Name of the TrOCR model.
        """
        self.processor = TrOCRProcessor.from_pretrained(trocr_model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def preprocess_image(self, image_path):
        """
        Load the image and perform pre-processing for segmentation.
        This function creates a binary image using adaptive thresholding,
        then applies a small morphological opening to reduce noise.
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            tuple: (binary_threshold_image, grayscale_image, original_color_image)
        """
        # Load image from disk (as BGR by default)
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path

        if image is None:
            raise ValueError("Could not load image")

        # Convert to grayscale for segmentation if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create a binary image using adaptive thresholding (inverted so text becomes white)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 10
        )

        # Apply a small morphological opening to remove noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return opening, gray, image  # binary image, grayscale, original color image

    def detect_text_lines(self, preprocessed_image, min_line_height=15, min_line_width=50):
        """
        Detect individual text lines in the binary image.
        Uses contour detection and hierarchical clustering based on y‐coordinate.
        
        Args:
            preprocessed_image (np.ndarray): Binary image after thresholding.
            min_line_height (int): Minimum height to be considered a text line.
            min_line_width (int): Minimum width to be considered a text line.
        
        Returns:
            list: List of bounding boxes for text lines as (y_min, y_max, x_min, x_max).
        """
        contours, _ = cv2.findContours(
            preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > min_line_height and w > min_line_width:
                boxes.append((x, y, w, h))
                
        # If no boxes found, return empty list.
        if not boxes:
            return []

        # If only one box is detected, skip clustering and return its bounding box.
        if len(boxes) < 2:
            x_min = min(box[0] for box in boxes)
            y_min = min(box[1] for box in boxes)
            x_max = max(box[0] + box[2] for box in boxes)
            y_max = max(box[1] + box[3] for box in boxes)
            return [(y_min, y_max, x_min, x_max)]

        # Extract y-centers for clustering
        y_centers = np.array([[y + h / 2] for x, y, w, h in boxes])
        # Perform hierarchical clustering to group boxes into lines
        Z = linkage(y_centers, 'ward')

        # Use median height to choose a clustering distance threshold
        heights = np.array([h for x, y, w, h in boxes])
        median_height = np.median(heights)
        max_dist = median_height * 0.7  # adjust threshold as needed

        clusters = fcluster(Z, max_dist, criterion='distance')
        # Group boxes by cluster
        lines_by_cluster = {}
        for box, cluster_id in zip(boxes, clusters):
            lines_by_cluster.setdefault(cluster_id, []).append(box)

        # Merge boxes of each cluster into a single line region
        line_regions = []
        for cluster_id, boxes_in_cluster in sorted(
            lines_by_cluster.items(), key=lambda x: min(b[1] for b in x[1])
        ):
            boxes_in_cluster.sort(key=lambda b: b[0])
            x_min = min(b[0] for b in boxes_in_cluster)
            y_min = min(b[1] for b in boxes_in_cluster)
            x_max = max(b[0] + b[2] for b in boxes_in_cluster)
            y_max = max(b[1] + b[3] for b in boxes_in_cluster)
            line_regions.append((y_min, y_max, x_min, x_max))
        return line_regions

    def detect_paragraphs(self, line_regions, original_gray):
        """
        Group the detected text lines into paragraphs using inter-line spacing.
        
        Args:
            line_regions (list): List of bounding boxes for text lines.
            original_gray (np.ndarray): Grayscale version of the original image.
        
        Returns:
            list: List of paragraphs, with each paragraph as a tuple
                  (paragraph_id, list of line regions, indentation).
        """
        if not line_regions:
            return []

        # Calculate spacing between successive lines
        line_spacing = []
        for i in range(1, len(line_regions)):
            prev_line_end = line_regions[i - 1][1]
            curr_line_start = line_regions[i][0]
            spacing = curr_line_start - prev_line_end
            line_spacing.append(spacing)
        if not line_spacing:
            return [(0, line_regions, line_regions[0][2])]
        mean_spacing = np.mean(line_spacing)
        std_spacing = np.std(line_spacing)
        paragraph_threshold = mean_spacing + 1.5 * std_spacing

        paragraphs = []
        current_para_lines = [line_regions[0]]
        current_para_id = 0
        first_line_indent = line_regions[0][2]

        for i in range(1, len(line_regions)):
            prev_line_end = line_regions[i - 1][1]
            curr_line_start = line_regions[i][0]
            spacing = curr_line_start - prev_line_end

            if spacing > paragraph_threshold:
                paragraphs.append((current_para_id, current_para_lines, first_line_indent))
                current_para_id += 1
                current_para_lines = [line_regions[i]]
                first_line_indent = line_regions[i][2]
            else:
                current_para_lines.append(line_regions[i])
        paragraphs.append((current_para_id, current_para_lines, first_line_indent))
        return paragraphs

    def extract_text_from_line(self, line_image):
        """
        Extract text from an image segment containing a single text line.
        Applies conversion to RGB (if needed) before passing to TrOCR.
        
        Args:
            line_image (np.ndarray): Cropped image of a text line.
        
        Returns:
            str: The text extracted from the line.
        """
        # Ensure the image has 3 channels
        if len(line_image.shape) == 2:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB)
        # Convert to PIL image for the model
        pil_image = Image.fromarray(line_image)
        # Get pixel values (the processor will do necessary resizing)
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def reconstruct_text(self, paragraphs, original_gray, color_image):
        """
        Reconstruct text using the detected paragraphs and lines.
        
        Args:
            paragraphs (list): List of paragraphs (paragraph_id, line_regions, indent).
            original_gray (np.ndarray): Grayscale image (for dimensions).
            color_image (np.ndarray): The original color image.
        
        Returns:
            str: The final extracted text with paragraph formatting.
        """
        full_text = []
        for para_id, line_regions, indent in paragraphs:
            paragraph_lines = []
            for y_min, y_max, x_min, x_max in line_regions:
                # Add padding to avoid clipping text
                y_min_pad = max(0, y_min - 2)
                y_max_pad = min(original_gray.shape[0], y_max + 2)
                x_min_pad = max(0, x_min - 2)
                x_max_pad = min(original_gray.shape[1], x_max + 2)
                line_image = color_image[y_min_pad:y_max_pad, x_min_pad:x_max_pad]
                if line_image.size == 0 or line_image.shape[0] == 0 or line_image.shape[1] == 0:
                    continue
                line_text = self.extract_text_from_line(line_image)
                paragraph_lines.append(line_text)
            paragraph_text = " ".join(paragraph_lines)
            full_text.append(paragraph_text)
        return "\n\n".join(full_text)

    def visualize_detection(self, image, line_regions, paragraphs):
        """
        Visualize detected text lines and paragraph boundaries on the image.
        
        Args:
            image (np.ndarray): The original image.
            line_regions (list): List of detected line bounding boxes.
            paragraphs (list): List of detected paragraph groups.
        """
        if len(image.shape) == 3:
            vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for y_min, y_max, x_min, x_max in line_regions:
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for para_id, lines, _ in paragraphs:
            color = colors[para_id % len(colors)]
            y_min = min(line[0] for line in lines)
            y_max = max(line[1] for line in lines)
            x_min = min(line[2] for line in lines)
            x_max = max(line[3] for line in lines)
            cv2.rectangle(vis_image, (x_min - 5, y_min - 5), (x_max + 5, y_max + 5), color, 3)

        plt.figure(figsize=(12, 12))
        plt.imshow(vis_image)
        plt.title("Detected Lines and Paragraphs")
        plt.axis('off')
        plt.show()

    def process_image(self, image_path, visualize=False):
        """
        Process a full-page image to detect text lines, group them into paragraphs,
        and extract text with structure.
        
        Args:
            image_path (str): Path to the input image.
            visualize (bool): Whether to show detection visualization.
            
        Returns:
            str: The extracted text with paragraph formatting.
        """
        preprocessed_image, original_gray, original_color = self.preprocess_image(image_path)
        line_regions = self.detect_text_lines(preprocessed_image)
        paragraphs = self.detect_paragraphs(line_regions, original_gray)
        if visualize:
            self.visualize_detection(original_color, line_regions, paragraphs)
        extracted_text = self.reconstruct_text(paragraphs, original_gray, original_color)
        return extracted_text

def main():
    extractor = HandwritingExtractor()
    # Replace with your full-page image path
    image_path = "page.png"
    text = extractor.process_image(image_path, visualize=True)
    print("Extracted Text with Paragraph Structure:")
    print("-" * 50)
    print(text)

if __name__ == "__main__":
    main()
