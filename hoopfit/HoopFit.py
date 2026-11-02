import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import tkinter as tk
from tkinter import messagebox, ttk
import os
import sys

# Configure modern UI theme colors
BG_COLOR = "#f5f5f5"
PRIMARY_COLOR = "#4a6fa5"
SECONDARY_COLOR = "#166088"
TEXT_COLOR = "#333333"
ENTRY_BG = "#ffffff"
BUTTON_HOVER = "#3a5a8a"

class BasketballPositionPredictor:
    def __init__(self):
        # Initialize models and scaler 初始化模型和标准化器 因为训练模型的数据标准化了所以新数据来也要标准化
        self.model_forward = None
        self.model_guard = None
        self.model_center = None
        self.scaler = None
        
        # Set up main application window
        self.root = tk.Tk()
        self.setup_window()
        self.create_widgets()
        
        # Load data and train models in background
        self.root.after(100, self.initialize_models)

    def setup_window(self):
        """Configure the main application window"""
        self.root.title("HoopFit")
        self.root.geometry("500x600")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)
        
        # Center the window on screen
        window_width = 500
        window_height = 600
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_x = int((screen_width - window_width) / 2)
        position_y = int((screen_height - window_height) / 2)
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

    def create_widgets(self):
        """Create all UI elements"""
        # Header frame
        header_frame = tk.Frame(self.root, bg=PRIMARY_COLOR)
        header_frame.pack(fill="x", pady=(0, 20))
        
        tk.Label(
            header_frame,
            text="Basketball Position Predictor",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg=PRIMARY_COLOR,
            pady=15
        ).pack()
        
        # Input frame
        input_frame = tk.Frame(self.root, bg=BG_COLOR)
        input_frame.pack(pady=10)
        
        # Stats entry fields
        self.stats_entries = {}
        stats_labels = [
            ("Games Played", "GP"),
            ("Points", "Pts"),
            ("Offensive Rebounds", "OReb"),
            ("Defensive Rebounds", "DReb"),
            ("Assists", "Ast"),
            ("Steals", "Stl"),
            ("Blocks", "Blk")
        ]
        
        for i, (label_text, stat_key) in enumerate(stats_labels):
            tk.Label(
                input_frame,
                text=label_text,
                font=("Segoe UI", 11),
                bg=BG_COLOR,
                fg=TEXT_COLOR
            ).grid(row=i, column=0, padx=10, pady=5, sticky="e")
            
            self.stats_entries[stat_key] = ttk.Entry(
                input_frame,
                font=("Segoe UI", 11),
                width=15
            )
            self.stats_entries[stat_key].grid(row=i, column=1, padx=10, pady=5)
        
        # Position selection
        tk.Label(
            input_frame,
            text="Select Position:",
            font=("Segoe UI", 11),
            bg=BG_COLOR,
            fg=TEXT_COLOR
        ).grid(row=len(stats_labels), column=0, padx=10, pady=10, sticky="e")
        
        self.position_var = tk.StringVar(value="Forward")
        self.position_menu = ttk.Combobox(
            input_frame,
            textvariable=self.position_var,
            values=["Forward", "Guard", "Center"],
            font=("Segoe UI", 11),
            state="readonly",
            width=13
        )
        self.position_menu.grid(row=len(stats_labels), column=1, padx=10, pady=10, sticky="w")
        
        # Predict button
        self.predict_button = tk.Button(
            self.root,
            text="PREDICT POSITION",
            font=("Segoe UI", 12, "bold"),
            bg=PRIMARY_COLOR,
            fg="white",
            activebackground=BUTTON_HOVER,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=20,
            pady=10,
            command=self.predict_position
        )
        self.predict_button.pack(pady=20)
        
        # Status label 看不出来效果因为来不及显示
        self.status_label = tk.Label(
            self.root,
            text="Initializing models...",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=SECONDARY_COLOR
        )
        self.status_label.pack(pady=10)

    def initialize_models(self):
        """Load data and train models"""
        try:
            self.status_label.config(text="Loading data...")
            self.root.update()
            
            # Load data
            data_path = os.path.join(self.get_base_dir(), "data.csv")
            data = pd.read_csv(data_path)
            
            features = ['GP', 'Pts', 'OReb', 'DReb', 'Ast', 'Stl', 'Blk']
            targets = ['F', 'G', 'C']
            X = data[features]
            
            # Split and scale data
            X_train, X_test, y_train_f, y_test_f = train_test_split(
                X, data[targets[0]], test_size=0.2, random_state=42)
            _, _, y_train_g, y_test_g = train_test_split(
                X, data[targets[1]], test_size=0.2, random_state=42)
            _, _, y_train_c, y_test_c = train_test_split(
                X, data[targets[2]], test_size=0.2, random_state=42)
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train models
            self.status_label.config(text="Training Forward model...")
            self.root.update()
            self.model_forward = LogisticRegression(C= 10, solver='lbfgs')
            self.model_forward.fit(X_train_scaled, y_train_f)
            
            self.status_label.config(text="Training Guard model...")
            self.root.update()
            self.model_guard = LogisticRegression(C= 10, solver='liblinear')
            self.model_guard.fit(X_train_scaled, y_train_g)
            
            self.status_label.config(text="Training Center model...")
            self.root.update()
            self.model_center = SVC(C=5, kernel='rbf', probability=True)
            self.model_center.fit(X_train_scaled, y_train_c)
            
            # Calculate accuracies
            X_test_scaled = self.scaler.transform(X_test)
            acc_f = roc_auc_score(y_test_f, self.model_forward.predict_proba(X_test_scaled)[:, 1])
            acc_g = roc_auc_score(y_test_g, self.model_guard.predict_proba(X_test_scaled)[:, 1])
            acc_c = roc_auc_score(y_test_c, self.model_center.predict_proba(X_test_scaled)[:, 1])
            # acc_c1 = roc_auc_score(y_train_c, self.model_center.predict_proba(X_train_scaled)[:, 1])
            # print(acc_c1)
            self.status_label.config(
                text=f"Models ready! (Classification Performance: F={acc_f:.1%}, G={acc_g:.1%}, C={acc_c:.1%})",
                fg="green"
            )
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            messagebox.showerror("Initialization Error", f"Failed to initialize models:\n{str(e)}")

    def predict_position(self):
        """Handle prediction button click"""
        try:
            # Validate models are ready
            if None in (self.model_forward, self.model_guard, self.model_center, self.scaler):
                messagebox.showwarning("Warning", "Models are still initializing. Please wait.")
                return
            
            # Get input values
            input_data = []
            for stat in ['GP', 'Pts', 'OReb', 'DReb', 'Ast', 'Stl', 'Blk']:
                try:
                    value = float(self.stats_entries[stat].get())
                    input_data.append(value)
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {stat}")
                    return
            
            # Scale input and predict
            input_scaled = self.scaler.transform([input_data])
            position = self.position_var.get()
            
            if position == "Forward":
                model = self.model_forward
            elif position == "Guard":
                model = self.model_guard
            else:
                model = self.model_center
            
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Show result
            result_window = tk.Toplevel(self.root)
            result_window.title("Prediction Result")
            result_window.geometry("400x200")
            result_window.configure(bg=BG_COLOR)
            result_window.resizable(False, False)
            
            # Center the result window
            window_width = 400
            window_height = 200
            screen_width = result_window.winfo_screenwidth()
            screen_height = result_window.winfo_screenheight()
            position_x = int((screen_width - window_width) / 2)
            position_y = int((screen_height - window_height) / 2)
            result_window.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
            
            # Result content
            tk.Label(
                result_window,
                text="PREDICTION RESULT",
                font=("Segoe UI", 14, "bold"),
                bg=BG_COLOR,
                fg=PRIMARY_COLOR
            ).pack(pady=(20, 10))
            
            result_text = f"Position: {position}\nSuitable: {'YES' if prediction == 1 else 'NO'}\nConfidence: {probability:.1%}"
            tk.Label(
                result_window,
                text=result_text,
                font=("Segoe UI", 12),
                bg=BG_COLOR,
                fg=TEXT_COLOR
            ).pack(pady=10)
            
            tk.Button(
                result_window,
                text="OK",
                command=result_window.destroy
            ).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{str(e)}")

    def get_base_dir(self):
        """Get the base directory path (works for both script and frozen exe)"""
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))

    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":
     # Run the application
    app = BasketballPositionPredictor()
    
    # Configure ttk style
    style = ttk.Style()
    style.configure("TCombobox", padding=5, relief="flat", background=ENTRY_BG)
    style.configure("TButton", padding=10, relief="flat", background=PRIMARY_COLOR, foreground="white")
    style.map("TButton", background=[("active", BUTTON_HOVER)])
    
    app.run()