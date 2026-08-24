from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import os

def apply_artistic_filter(image_path, output_path="artistic_image.png"):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_contrast = ImageEnhance.Contrast(img_resized).enhance(1.5)
        img_colorful = ImageEnhance.Color(img_contrast).enhance(1.5)
        img_artistic = ImageEnhance.Sharpness(img_colorful).enhance(1.5)

        plt.imshow(img_artistic)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Artistic image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Artistic Image Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        # derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_artistic{ext}"
        apply_artistic_filter(image_path, output_file)