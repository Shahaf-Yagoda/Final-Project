#!/usr/bin/env python3
"""
Simple camera test to verify camera access and release
"""
import cv2
import time

def test_camera_access():
    """Test basic camera access and release"""
    print("Testing camera access...")
    
    # Test camera initialization
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available")
        return False
    
    print("✅ Camera opened successfully")
    
    # Set camera properties for better compatibility
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("✅ Camera properties set")
    
    # Test frame capture
    success, frame = cap.read()
    if not success:
        print("❌ Could not read frame from camera")
        cap.release()
        return False
    
    print(f"✅ Frame captured successfully: {frame.shape}")
    
    # Test multiple captures
    for i in range(5):
        success, frame = cap.read()
        if not success:
            print(f"❌ Failed to read frame {i+1}")
            cap.release()
            return False
        print(f"✅ Frame {i+1} captured")
        time.sleep(0.1)
    
    # Test camera release
    cap.release()
    print("✅ Camera released successfully")
    
    # Test re-initialization after release
    time.sleep(1)
    print("Testing camera re-initialization...")
    cap2 = cv2.VideoCapture(0)
    if not cap2.isOpened():
        print("❌ Could not re-open camera after release")
        return False
    
    print("✅ Camera re-opened successfully")
    cap2.release()
    print("✅ Camera released again")
    
    return True

if __name__ == "__main__":
    result = test_camera_access()
    if result:
        print("\n✅ Camera test passed - camera is working properly")
    else:
        print("\n❌ Camera test failed - there may be camera issues")
    exit(0 if result else 1)