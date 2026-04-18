import ollama
import cv2
import base64
import numpy as np

def test_ollama_moondream():
    print("Testing Ollama with Moondream...")
    
    # Create a dummy image (100x100 green square)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [0, 255, 0]
    
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    prompt = "What color is this image?"
    
    try:
        response = ollama.chat(
            model='moondream',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [img_base64]
            }]
        )
        print("Response:", response['message']['content'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ollama_moondream()
