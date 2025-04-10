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
        Initialize the handwriting extractor with TrOCR model.

        Args:
            trocr_model_name (str): Name of the TrOCR model to use
        """
        self.processor = TrOCRProcessor.from_pretrained(trocr_model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def preprocess_image(self, image_path):
        """
        Load and preprocess the handwritten image.

        Args:
            image_path (str): Path to the input image

        Returns:
            np.ndarray: Preprocessed image
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path

        if image is None:
            raise ValueError("Could not load image")

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding to highlight text
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10
        )

        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return opening, gray, image  # Return original image too

    def detect_text_lines(self, preprocessed_image, min_line_height=15, min_line_width=50):
        """
        Detect individual text lines from the preprocessed image.

        Args:
            preprocessed_image (np.ndarray): Preprocessed binary image
            min_line_height (int): Minimum height for a valid text line
            min_line_width (int): Minimum width for a valid text line

        Returns:
            list: List of detected line regions (y_start, y_end, x_start, x_end, line_image)
        """
        # Find contours to detect text regions
        contours, _ = cv2.findContours(
            preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get bounding boxes for each contour
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > min_line_height and w > min_line_width:  # Filter small noise contours
                boxes.append((x, y, w, h))

        # If no boxes detected, return empty list
        if not boxes:
            return []

        # Extract y-coordinates and heights for clustering
        y_centers = np.array([[y + h/2] for x, y, w, h in boxes])

        # Perform hierarchical clustering to group contours into lines
        Z = linkage(y_centers, 'ward')

        # Determine optimal number of clusters based on line height distribution
        heights = np.array([h for _, _, _, h in boxes])
        median_height = np.median(heights)
        max_dist = median_height * 0.7  # Adjust this threshold based on writing style

        # Cluster the lines
        clusters = fcluster(Z, max_dist, criterion='distance')

        # Group boxes by cluster
        lines_by_cluster = {}
        for box, cluster_id in zip(boxes, clusters):
            x, y, w, h = box
            if cluster_id not in lines_by_cluster:
                lines_by_cluster[cluster_id] = []
            lines_by_cluster[cluster_id].append((x, y, w, h))

        # Sort clusters by y-coordinate (top to bottom)
        line_regions = []
        for cluster_id, boxes in sorted(lines_by_cluster.items(),
                                        key=lambda x: min(box[1] for box in x[1])):
            # Sort boxes by x-coordinate (left to right)
            boxes.sort(key=lambda box: box[0])

            # Get the combined bounding box for the line
            x_min = min(box[0] for box in boxes)
            y_min = min(box[1] for box in boxes)
            x_max = max(box[0] + box[2] for box in boxes)
            y_max = max(box[1] + box[3] for box in boxes)

            line_regions.append((y_min, y_max, x_min, x_max))

        return line_regions

    def detect_paragraphs(self, line_regions, original_image):
        """
        Group text lines into paragraphs based on spacing and indentation.

        Args:
            line_regions (list): List of line regions (y_start, y_end, x_start, x_end)
            original_image (np.ndarray): Original grayscale image

        Returns:
            list: List of paragraph information (para_id, line_regions, indentation)
        """
        if not line_regions:
            return []

        # Calculate inter-line spacing
        line_spacing = []
        for i in range(1, len(line_regions)):
            prev_line_end = line_regions[i-1][1]  # y_end of previous line
            curr_line_start = line_regions[i][0]  # y_start of current line
            spacing = curr_line_start - prev_line_end
            line_spacing.append(spacing)

        # Calculate mean and standard deviation of line spacing
        if not line_spacing:
            return [(0, line_regions, 0)]  # Single paragraph

        mean_spacing = np.mean(line_spacing)
        std_spacing = np.std(line_spacing)

        # Define threshold for paragraph break (larger spacing)
        paragraph_threshold = mean_spacing + 1.5 * std_spacing

        # Group lines into paragraphs
        paragraphs = []
        current_para_lines = [line_regions[0]]
        current_para_id = 0

        # Check indentation for first line
        first_line_indent = line_regions[0][2]  # x_start of first line

        for i in range(1, len(line_regions)):
            prev_line_end = line_regions[i-1][1]
            curr_line_start = line_regions[i][0]
            spacing = curr_line_start - prev_line_end

            # New paragraph detected if spacing is larger than threshold
            if spacing > paragraph_threshold:
                # Check indentation for additional paragraph indicators
                curr_indent = line_regions[i][2]
                paragraphs.append((current_para_id, current_para_lines, first_line_indent))
                current_para_id += 1
                current_para_lines = [line_regions[i]]
                first_line_indent = curr_indent
            else:
                current_para_lines.append(line_regions[i])

        # Add the last paragraph
        paragraphs.append((current_para_id, current_para_lines, first_line_indent))

        return paragraphs

    def extract_text_from_line(self, line_image):
        """
        Extract text from a line image using TrOCR model.

        Args:
            line_image (np.ndarray): Image of a text line

        Returns:
            str: Extracted text
        """
        # Convert grayscale to RGB if needed
        if len(line_image.shape) == 2:
            # Convert grayscale to RGB
            line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(line_image)

        # Preprocess image for TrOCR
        pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)

        # Generate text
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

    def reconstruct_text(self, paragraphs, original_image, color_image):
        """
        Reconstruct the text with paragraph structure.

        Args:
            paragraphs (list): List of paragraph information
            original_image (np.ndarray): Original grayscale image
            color_image (np.ndarray): Original color image

        Returns:
            str: Reconstructed text with paragraph formatting
        """
        full_text = []

        for para_id, line_regions, indent in paragraphs:
            paragraph_lines = []

            for y_min, y_max, x_min, x_max in line_regions:
                # Add padding to ensure we don't crop text too tightly
                y_min_pad = max(0, y_min - 2)
                y_max_pad = min(original_image.shape[0], y_max + 2)
                x_min_pad = max(0, x_min - 2)
                x_max_pad = min(original_image.shape[1], x_max + 2)

                # Extract the line from the color image
                line_image = color_image[y_min_pad:y_max_pad, x_min_pad:x_max_pad]

                # Skip empty or invalid lines
                if line_image.size == 0 or line_image.shape[0] == 0 or line_image.shape[1] == 0:
                    continue

                # Extract text from the line
                line_text = self.extract_text_from_line(line_image)
                paragraph_lines.append(line_text)

            # Join the lines in the paragraph
            paragraph_text = " ".join(paragraph_lines)
            full_text.append(paragraph_text)

        # Join paragraphs with double new lines
        return "\n\n".join(full_text)

    def process_image(self, image_path, visualize=False):
        """
        Process a handwritten image and extract structured text.

        Args:
            image_path (str): Path to the input image
            visualize (bool): Whether to visualize the detected lines and paragraphs

        Returns:
            str: Extracted text with paragraph structure
        """
        # Preprocess the image
        preprocessed_image, original_gray, original_color = self.preprocess_image(image_path)

        # Detect text lines
        line_regions = self.detect_text_lines(preprocessed_image)

        # Group lines into paragraphs
        paragraphs = self.detect_paragraphs(line_regions, original_gray)

        # Visualize if requested
        if visualize:
            self.visualize_detection(original_color, line_regions, paragraphs)

        # Reconstruct text
        return self.reconstruct_text(paragraphs, original_gray, original_color)

    def visualize_detection(self, image, line_regions, paragraphs):
        """
        Visualize the detected lines and paragraphs.

        Args:
            image (np.ndarray): Original image
            line_regions (list): List of line regions
            paragraphs (list): List of paragraph information
        """
        # Convert to RGB for visualization
        if len(image.shape) == 3:
            vis_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Draw line boundaries
        for y_min, y_max, x_min, x_max in line_regions:
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Draw paragraph boundaries with different colors
        colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for para_id, lines, _ in paragraphs:
            color = colors[para_id % len(colors)]
            y_min = min(line[0] for line in lines)
            y_max = max(line[1] for line in lines)
            x_min = min(line[2] for line in lines)
            x_max = max(line[3] for line in lines)
            cv2.rectangle(vis_image, (x_min-5, y_min-5), (x_max+5, y_max+5), color, 3)

        plt.figure(figsize=(12, 12))
        plt.imshow(vis_image)
        plt.title("Detected Lines and Paragraphs")
        plt.axis('off')
        plt.show()

# Example usage
def main():
    # Initialize the extractor
    extractor = HandwritingExtractor()

    # Process an image
    image_path = "page.png"
    extracted_text = extractor.process_image(image_path, visualize=True)

    print("Extracted Text with Paragraph Structure:")
    print("-" * 50)
    print(extracted_text)

if __name__ == "__main__":
    main()