"""Script to generate gRPC Python code from proto file."""
import subprocess
import sys
import os

def generate_grpc_code():
    """Generate gRPC Python code from proto file."""
    proto_file = "killfeed_detection.proto"
    
    if not os.path.exists(proto_file):
        print(f"❌ Proto file not found: {proto_file}")
        return False
    
    try:
        # Generate gRPC code
        cmd = [
            sys.executable, '-m', 'grpc_tools.protoc',
            '-I.',  # Current directory
            '--python_out=.',
            '--grpc_python_out=.',
            proto_file
        ]
        
        print(f"🔧 Generating gRPC code from {proto_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ gRPC code generated successfully!")
            print("   Generated files:")
            print("   - killfeed_detection_pb2.py")
            print("   - killfeed_detection_pb2_grpc.py")
            return True
        else:
            print(f"❌ Failed to generate gRPC code:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error generating gRPC code: {e}")
        return False

if __name__ == '__main__':
    success = generate_grpc_code()
    sys.exit(0 if success else 1)

