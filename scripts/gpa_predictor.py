import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


class GPAPredictorApp:
    def __init__(self, model_path):
        self.model_path = model_path
        self.pipeline = None
        self.input_data = {}

    def load_pipeline(self):
        """Load the trained pipeline."""
        try:
            # Load the model
            self.pipeline = joblib.load(self.model_path)
            if not hasattr(self.pipeline, 'predict'):
                st.error("The loaded object is not a valid pipeline. Please check the `.pkl` file.")
                st.stop()
        except FileNotFoundError:
            st.error(f"Model file not found at {self.model_path}. Please check the path.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    def configure_page(self):
        """Set the Streamlit page configuration."""
        st.set_page_config(
            page_title="GPA Predictor",
            page_icon="🎓",
            layout="centered",
            initial_sidebar_state="expanded",
        )

    def get_user_inputs(self):
        """Collect inputs from the user."""
        st.header("Input Features")

        # Binary variables
        demo_race = st.radio("Underrepresented Race (No = 1, Yes = 0)", [1, 0])
        demo_gender = st.radio("Gender (Male = 0, Female = 1)", [0, 1])
        demo_firstgen = st.radio("First-Generation Student (No = 0, Yes = 1)", [0, 1])

        # Continuous variables
        TotalSleepTime = st.slider("Total Sleep Time (minutes)", 194.78, 587.67, 397.37)
        midpoint_sleep = st.slider("Midpoint Sleep (minutes after 11 pm)", 247.07, 724.67, 398.70)
        frac_nights_with_data = st.slider("Fraction of Nights with Data", 0.21, 1.0, 0.87)
        daytime_sleep = st.slider("Daytime Sleep (minutes)", 2.27, 292.30, 41.28)
        cum_gpa = st.slider("Cumulative GPA", 1.21, 4.0, 3.47)
        Zterm_units_ZofZ = st.slider("Z-term Units ZofZ", -3.98, 4.06, -0.0029)

        # Bedtime MSSD
        st.markdown("**Bedtime MSSD (Enter 3 values separated by commas)**")
        bedtime_mssd_input = st.text_input("Enter values (e.g., 0.45, 0.14, 0.13)", "0.45, 0.14, 0.13")
        try:
            bedtime_mssd = [float(x.strip()) for x in bedtime_mssd_input.split(",")]
            if len(bedtime_mssd) != 3:
                raise ValueError("Please enter exactly 3 values.")
        except ValueError as e:
            st.error(f"Invalid input for bedtime MSSD: {e}")
            st.stop()

        # Save inputs as a dictionary
        self.input_data = {
            'demo_race': demo_race,
            'demo_gender': demo_gender,
            'demo_firstgen': demo_firstgen,
            'bedtime_mssd': np.mean(bedtime_mssd),
            'TotalSleepTime': TotalSleepTime,
            'midpoint_sleep': midpoint_sleep,
            'frac_nights_with_data': frac_nights_with_data,
            'daytime_sleep': daytime_sleep,
            'cum_gpa': cum_gpa,
            'Zterm_units_ZofZ': Zterm_units_ZofZ,
        }

    def display_input_summary(self):
        """Display the summary of the user inputs."""
        st.subheader("Input Data Summary")
        input_df = pd.DataFrame(self.input_data, index=[0])

        # Remove the index column and display the dataframe
        st.write("**Section 1:**")
        st.dataframe(input_df[['demo_race', 'demo_gender', 'demo_firstgen', 'bedtime_mssd', 'TotalSleepTime']])
        st.write("**Section 2:**")
        st.dataframe(input_df[['midpoint_sleep', 'frac_nights_with_data', 'daytime_sleep', 'cum_gpa', 'Zterm_units_ZofZ']])

    def predict_gpa(self):
        """Make GPA predictions using the loaded pipeline."""
        if not self.input_data:
            st.error("No input data available for prediction!")
            return

        input_df = pd.DataFrame(self.input_data, index=[0])
        try:
            prediction = self.pipeline.predict(input_df)
            st.markdown(
                f"<p style='color: #78C2AD; font-size: 20px;'>Predicted Term GPA: {prediction[0]:.2f}</p>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    def run(self):
        """Run the Streamlit app."""
        self.configure_page()
        self.load_pipeline()

        self.get_user_inputs()
        self.display_input_summary()

        if st.button("Predict Term GPA"):
            self.predict_gpa()


# Run the app
if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the model file
    model_path = os.path.join(script_dir, "..", "models", "best_random_forest_model.pkl")

    app = GPAPredictorApp(model_path)
    app.run()
