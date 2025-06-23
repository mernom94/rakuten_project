import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200



def test_training_status():
    """Test the training status endpoint"""
    print("\nTesting training status endpoint...")
    response = requests.get(f"{BASE_URL}/training/status")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def main():
    """Run all tests"""
    print("=== Testing ML API ===")
    
    tests = [
        ("Health Check", test_health_check),
        ("Training Status", test_training_status),
        # Add more tests here as needed
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Error in {test_name}: {str(e)}")
            results[test_name] = False
        
        time.sleep(2)  # Small delay between tests
    
    print("\n=== Test Results ===")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main()