"""Quick test to see if OCR will work in a fresh Python process."""
print("Testing OCR in a fresh Python process...")
print("="*60)

try:
    print("1. Importing paddleocr...")
    import paddleocr
    print("   ✅ Import successful")
    
    print("2. Initializing PaddleOCR...")
    import logging
    logging.getLogger('ppocr').setLevel(logging.WARNING)
    
    ocr = paddleocr.PaddleOCR(lang='en')
    print("   ✅ Initialization successful")
    
    print("3. Testing FreeFireTextDetector...")
    from text import FreeFireTextDetector
    detector = FreeFireTextDetector()
    print("   ✅ FreeFireTextDetector works!")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! OCR is working!")
    print("="*60)
    print("\n💡 If your main script still fails:")
    print("   1. CLOSE all Python terminals/IDEs")
    print("   2. RESTART your computer")
    print("   3. Open a NEW terminal")
    print("   4. Run: python ffkillblock.py")
    
except Exception as e:
    error_msg = str(e)
    if "shm.dll" in error_msg or "WinError 127" in error_msg:
        print("\n" + "="*60)
        print("❌ Visual C++ Redistributables still not working")
        print("="*60)
        print("\nEven in a fresh process, the error persists.")
        print("\n📋 Try these steps:")
        print("   1. Make sure Visual C++ Redistributables is installed:")
        print("      - Go to: Control Panel → Programs → Programs and Features")
        print("      - Look for: 'Microsoft Visual C++ 2015-2022 Redistributable (x64)'")
        print("   2. If not found, download and install:")
        print("      https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("   3. RESTART your computer (very important!)")
        print("   4. Run this test again")
        print("\n   Error:", error_msg)
    else:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


