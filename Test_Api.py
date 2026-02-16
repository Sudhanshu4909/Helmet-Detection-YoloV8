
import requests
import json

# CORRECT URL FORMAT
API_URL = "https://sudhanshu03-helmet-detection.hf.space"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Test 1: Health Check")
    print("="*60)
    
    url = f"{API_URL}/health"
    print(f"Testing: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root():
    """Test root endpoint"""
    print("\n" + "="*60)
    print("Test 2: Root Endpoint")
    print("="*60)
    
    url = f"{API_URL}/"
    print(f"Testing: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict(image_path):
    """Test image prediction"""
    print("\n" + "="*60)
    print("Test 3: Image Prediction")
    print("="*60)
    
    url = f"{API_URL}/predict"
    print(f"Testing: {url}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'conf_threshold': 0.25}
            
            response = requests.post(url, files=files, params=params, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data['success']}")
            print(f"Detections: {data['count']}")
            
            for i, det in enumerate(data['detections'], 1):
                print(f"\nDetection {i}:")
                print(f"  Class: {det['class_name']}")
                print(f"  Confidence: {det['confidence']:.4f}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("="*60)
    print("Helmet Detection API - Fixed Test Suite")
    print("="*60)
    print(f"API URL: {API_URL}")
    print("="*60)
    
    # Test 1: Health
    health_ok = test_health()
    
    # Test 2: Root
    root_ok = test_root()
    
    # Test 3: Prediction (if image provided)
    import sys
    if len(sys.argv) > 1:
        predict_ok = test_predict(sys.argv[1])
    else:
        print("\nSkipping prediction test (no image provided)")
        print("Usage: python test_api_fixed.py <image_path>")
        predict_ok = None
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Health Check: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"Root Endpoint: {'✓ PASS' if root_ok else '✗ FAIL'}")
    if predict_ok is not None:
        print(f"Prediction: {'✓ PASS' if predict_ok else '✗ FAIL'}")
    
    # Print curl examples
    print("\n" + "="*60)
    print("cURL Examples")
    print("="*60)
    
    print(f"\n1. Health Check:")
    print(f"curl {API_URL}/health")
    
    print(f"\n2. Prediction:")
    print(f"""curl -X POST "{API_URL}/predict" \\
  -F "file=@image.jpg" \\
  -F "conf_threshold=0.25\"""")

if __name__ == "__main__":
    main()