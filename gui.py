import tkinter as tk
from tkinter import ttk, messagebox
import requests
import json
import joblib
from PIL import Image, ImageTk


class CO2PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CO2 Emissions Predictor")
        self.root.geometry("500x700")
        self.root.configure(bg="#f0f2f5")

        # Load label encoders for dropdowns
        self.label_encoders = joblib.load("models/label_encoders.pkl")
        self.mappings = {key: le.classes_.tolist()
                         for key, le in self.label_encoders.items()}

        # API endpoint
        self.api_url = "http://127.0.0.1:8001/predict"

        # Create main frame
        self.main_frame = tk.Frame(self.root, bg="#f0f2f5")
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Title
        tk.Label(
            self.main_frame, text="CO2 Emissions Predictor", font=("Helvetica", 20, "bold"),
            bg="#f0f2f5", fg="#333"
        ).pack(pady=10)

        # Input fields
        self.entries = {}
        self.labels = [
            ("Make", "combobox", self.mappings["Make"]),
            # Limit for performance
            ("Model", "combobox", self.mappings["Model"][:50]),
            ("Vehicle Class", "combobox", self.mappings["Vehicle Class"]),
            ("Engine Size (L)", "entry", None),
            ("Transmission", "combobox", self.mappings["Transmission"]),
            ("Fuel Type", "combobox", self.mappings["Fuel Type"]),
            ("Fuel Consumption Hwy (L/100 km)", "entry", None)
        ]

        for i, (label, input_type, values) in enumerate(self.labels):
            tk.Label(
                self.main_frame, text=label, font=("Helvetica", 12), bg="#f0f2f5", fg="#555"
            ).pack(anchor="w", padx=10, pady=5)

            if input_type == "combobox":
                entry = ttk.Combobox(self.main_frame, values=values, font=(
                    "Helvetica", 12), state="readonly")
                entry.pack(fill="x", padx=10, pady=5)
                entry.set(values[0])
            else:
                entry = tk.Entry(self.main_frame, font=(
                    "Helvetica", 12), bd=2, relief="flat", bg="#fff")
                entry.pack(fill="x", padx=10, pady=5)

            self.entries[label] = entry

        # Predict button with animation
        self.predict_button = tk.Button(
            self.main_frame, text="Predict", font=("Helvetica", 14, "bold"),
            bg="#4CAF50", fg="white", bd=0, relief="flat", command=self.animate_predict
        )
        self.predict_button.pack(pady=20, ipadx=20, ipady=10)

        # Result label
        self.result_label = tk.Label(
            self.main_frame, text="", font=("Helvetica", 12), bg="#f0f2f5", fg="#333333", wraplength=400
        )
        self.result_label.pack(pady=10)

        # Animation variables
        self.fade_alpha = 0

    def animate_predict(self):
        self.predict_button.config(state="disabled", bg="#45a049")
        self.fade_alpha = 0
        self.result_label.config(text="")
        self.root.after(100, self.fade_in_result)

        # Call predict after animation starts
        self.root.after(200, self.predict)

    def fade_in_result(self):
        if self.fade_alpha < 1:
            self.fade_alpha += 0.1
            brightness = int(200 - self.fade_alpha * 150)
            color = f"#{brightness:02x}{brightness:02x}{brightness:02x}"
            self.result_label.config(fg=color)
            self.root.after(50, self.fade_in_result)
        else:
            self.predict_button.config(state="normal", bg="#4CAF50")

    def predict(self):
        try:
            # Collect input data
            input_data = {}
            for label, entry in self.entries.items():
                if label in ["Engine Size (L)", "Fuel Consumption Hwy (L/100 km)"]:
                    # Ensure numerical fields are float
                    key = label.replace(" (L)", "_L").replace(
                        " (L/100 km)", "_L_100km").replace(" ", "_")
                    input_data[key] = float(entry.get())
                else:
                    # Categorical fields (from combobox)
                    value = entry.get()
                    key = label.replace(" ", "_")
                    encoded_value = self.label_encoders[label].transform([value])[
                        0]
                    input_data[key] = int(encoded_value)

            # Send request to FastAPI
            response = requests.post(self.api_url, json=input_data)
            response.raise_for_status()
            prediction = response.json()["prediction"]

            # Display prediction with animation
            self.result_label.config(
                text=f"Predicted CO2 Emissions: {prediction:.2f} g/km")
        except ValueError:
            messagebox.showerror(
                "خطأ في المدخلات", "الرجاء إدخال قيم عددية صحيحة.")
        except requests.RequestException as e:
            messagebox.showerror(
                "خطأ في الـ API", f"Unprocessable Entity: {str(e)}")
        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CO2PredictionApp(root)
    root.mainloop()

