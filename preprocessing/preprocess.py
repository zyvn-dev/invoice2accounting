import cv2
import os

def preprocess_image(input_path: str, output_path: str) -> None:
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")

    original_dims = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced)

    print(f"Original dimensions: {original_dims}")
    print(f"Output dimensions:   {enhanced.shape[:2]}")

if __name__ == "__main__":
    preprocess_image(
        input_path="data/samples/sample1.jpg",
        output_path="data/processed/sample1_clean.jpg",
    )