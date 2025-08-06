import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load job database
jobs = pd.DataFrame({
    "title": ["Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer", "Data Analyst"],
    "location": ["New York", "San Francisco", "New York", "San Francisco", "Chicago"],
    "skills": ["python, java, c++", "python, r, machine learning", "product management, marketing", "aws, docker, kubernetes", "excel, sql, data analysis"]
})

class JobPredictionApp:
    def __init__(self):
        # Initialize the main window and configure its appearance
        self.window = tk.Tk()
        self.window.title("Job Prediction App")
        self.window.configure(bg='yellow')

        # Set the window size and center it on the screen
        window_width = 800
        window_height = 600
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.window.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        # Create a main frame to hold all widgets and center it
        self.main_frame = tk.Frame(self.window, padx=20, pady=20, bg='yellow')
        self.main_frame.pack(expand=True)

        # Add the heading label with a large red font
        self.heading_label = tk.Label(self.main_frame, 
                                      text="JOB PREDICTION APP BY PIXEL PIONEERS", 
                                      font=("Victor Mono", 24, "bold"), # Increased font size
                                      bg='yellow',
                                      fg='blue') # Set font color to red
        self.heading_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Create form fields
        self.location_label = tk.Label(self.main_frame, text="Location:", bg='yellow')
        self.location_entry = tk.Entry(self.main_frame)

        self.skills_label = tk.Label(self.main_frame, text="Skills:", bg='yellow')
        self.skills_entry = tk.Entry(self.main_frame)

        self.predict_button = tk.Button(self.main_frame, text="Predict Job", command=self.predict_job)

        # Layout form fields and buttons
        self.location_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.location_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.skills_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.skills_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.predict_button.grid(row=3, column=0, columnspan=2, padx=5, pady=20)

        # Make the columns expandable to push the content to the center
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

    def predict_job(self):
        # Get user input
        location = self.location_entry.get().strip().title()
        skills = self.skills_entry.get().strip().lower()

        # Handle empty input fields
        if not location and not skills:
            messagebox.showinfo("Input Required", "Please enter a location and your skills.")
            return
        if not location:
            messagebox.showinfo("Location Required", "Please enter a location.")
            return
        if not skills:
            messagebox.showinfo("Skills Required", "Please enter your skills to get a prediction.")
            return

        # Filter jobs based on location
        location_jobs = jobs[jobs["location"].str.lower() == location.lower()]
        
        # Handle the case where no jobs match the location
        if location_jobs.empty:
            messagebox.showinfo("No Jobs Found", f"Sorry, we couldn't find any jobs in {location}.")
            return

        # Calculate similarity between user skills and job skills
        vectorizer = TfidfVectorizer()
        job_skills = location_jobs["skills"].tolist()
        
        # Ensure job_skills is not empty before fitting the vectorizer
        if not job_skills:
            messagebox.showinfo("No Skills Found", "The jobs in this location do not have any listed skills.")
            return

        tfidf = vectorizer.fit_transform(job_skills)
        user_skills_tfidf = vectorizer.transform([skills])
        similarity = cosine_similarity(tfidf, user_skills_tfidf)

        # Get top matching job
        top_job_index = similarity.argmax()
        top_job = location_jobs.iloc[top_job_index]

        # Call the new function to display the result in a larger window
        self.show_prediction_window(top_job['title'])

    def show_prediction_window(self, job_title):
        """Creates a new top-level window to display the prediction in a larger font."""
        prediction_window = tk.Toplevel(self.window)
        prediction_window.title("Predicted Job")
        prediction_window.geometry("500x150")
        prediction_window.configure(bg='yellow')
        
        # Make the window centered relative to the main window
        x = self.window.winfo_x() + self.window.winfo_width() // 2 - 250
        y = self.window.winfo_y() + self.window.winfo_height() // 2 - 75
        prediction_window.geometry(f"+{x}+{y}")
        
        # Use a large, bold font for the output label
        output_text = f"Based on your skills and location, we recommend a career as a {job_title}."
        output_label = tk.Label(prediction_window, 
                                text=output_text, 
                                font=("Helvetica", 14, "bold"),
                                bg='yellow', 
                                wraplength=450)
        output_label.pack(expand=True, padx=20, pady=20)
        
        # Add a close button
        close_button = tk.Button(prediction_window, text="Close", command=prediction_window.destroy)
        close_button.pack(pady=(0, 10))

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = JobPredictionApp()
    app.run()
