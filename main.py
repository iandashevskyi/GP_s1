import subprocess


def main():
    print("Select mode:")
    print("1. Train q-learning")
    print("2. Train sarsa")
    print("3. Train double q-learning")
    print("4. Test agent")
    print("5. Play against agent")

    choice = input("Enter your choice: ")

    if choice == "1":
        subprocess.run(["python", "train_q.py"])
    elif choice == "2":
        subprocess.run(["python", "train_sarsa.py"])
    elif choice == "3":
        subprocess.run(["python", "train_double_q.py"])
    elif choice == "4":
        subprocess.run(["python", "evaluate.py"])
    elif choice == "5":
        subprocess.run(["python", "play_human.py"])
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
