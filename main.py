"""
main.py — CLI entry point for the Real-Time Facial Recognition System.

Opens the webcam ONCE at startup. Provides:
  1) Start (unified recognize + auto-register)
  2) List registered users
  3) Delete a user
  4) Clear all users
  5) Exit
"""

import sys
import os
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pre-load all modules at startup
from database import init_db, get_all_users, delete_user
from face_engine import init_face_app
from recognize import start_unified


def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 60)
    print("  🔍  REAL-TIME FACIAL RECOGNITION SYSTEM")
    print("  Engine: InsightFace + AuraFace-v1")
    print("  Mode: Auto Recognize + Auto Register")
    print("=" * 60)


def print_menu():
    """Print the main menu."""
    print("\n┌─────────────────────────────────────────┐")
    print("│              MAIN MENU                  │")
    print("├─────────────────────────────────────────┤")
    print("│  1) Start (Recognize + Auto-Register)   │")
    print("│  2) List Registered Users               │")
    print("│  3) Delete a User                       │")
    print("│  4) Clear All Users                     │")
    print("│  5) Exit                                │")
    print("└─────────────────────────────────────────┘")


def list_users():
    """Display all registered users."""
    users = get_all_users()
    if not users:
        print("\n  No registered users found.")
        return

    print(f"\n  Registered Users ({len(users)}):")
    print(f"  {'ID':<12} {'Registered':<25}")
    print(f"  {'-'*12} {'-'*25}")
    for user_id, created_at in users:
        print(f"  {user_id:<12} {created_at:<25}")


def delete_user_menu():
    """Interactive menu to delete a user."""
    users = get_all_users()
    if not users:
        print("\n  No registered users to delete.")
        return

    list_users()
    user_id = input("\n  Enter user ID to delete (or 'cancel'): ").strip()

    if user_id.lower() == "cancel":
        return

    # Check user exists
    found = any(uid == user_id for uid, _ in users)
    if not found:
        print(f"  User ID '{user_id}' not found!")
        return

    confirm = input(f"  Delete user '{user_id}'? (y/n): ").strip().lower()
    if confirm == "y":
        delete_user(user_id)
        face_path = os.path.join("data", "registered_faces", f"{user_id}.jpg")
        if os.path.exists(face_path):
            os.remove(face_path)
        print(f"  User '{user_id}' deleted.")
    else:
        print("  Cancelled.")


def clear_all_users():
    """Delete all users from the database."""
    users = get_all_users()
    if not users:
        print("\n  No users to clear.")
        return

    confirm = input(f"\n  Delete ALL {len(users)} users? This cannot be undone. (yes/no): ").strip().lower()
    if confirm == "yes":
        for uid, _ in users:
            delete_user(uid)
            face_path = os.path.join("data", "registered_faces", f"{uid}.jpg")
            if os.path.exists(face_path):
                os.remove(face_path)
        print(f"  Cleared {len(users)} user(s).")
    else:
        print("  Cancelled.")


def main():
    """Main entry point."""
    print_banner()

    # Step 1: Initialize database
    print("\n[Init] Setting up database...")
    init_db()

    # Step 2: Initialize face engine
    print("[Init] Loading face recognition engine...")
    app = init_face_app()

    # Step 3: Open webcam ONCE
    print("[Init] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Init] ERROR: Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[Init] Webcam ready!")

    print("\n[Init] System ready!\n")

    try:
        while True:
            print_menu()
            choice = input("\n  Select option (1-5): ").strip()

            if choice == "1":
                start_unified(app, cap)

            elif choice == "2":
                list_users()

            elif choice == "3":
                delete_user_menu()

            elif choice == "4":
                clear_all_users()

            elif choice == "5":
                print("\n  Goodbye! 👋\n")
                break

            else:
                print("  Invalid option. Please select 1-5.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Cleanup] Webcam released.")


if __name__ == "__main__":
    main()
