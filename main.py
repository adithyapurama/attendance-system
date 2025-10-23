import os
import sys
import subprocess

def run_registration():
    print("🚀 Starting Registration Service (port 5000)...")
    os.environ["FLASK_ENV"] = "production"
    subprocess.run([sys.executable, "registration.py"])

def run_verification():
    print("🧠 Starting Attendance Verification Service (port 5000)...")
    os.environ["FLASK_ENV"] = "production"
    subprocess.run([sys.executable, "attendance_verification.py"])

def show_help():
    print("""
Usage:
    python main.py [service]

Services:
    registration     Start registration service (admin_add_user)
    verify           Start attendance verification service (verify_attendance)
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(0)

    service = sys.argv[1].lower()

    if service == "registration":
        run_registration()
    elif service in ("verify", "verification"):
        run_verification()
    else:
        show_help()
