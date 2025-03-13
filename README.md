# CSV Question Answering and Visualization Application

This project is a Gradio-based application that allows users to upload a CSV file, ask questions about the data, and visualize insights interactively. It is designed to run on **Google Colab** for ease of use.

## Getting Started on Google Colab

### 1. Open Google Colab
- Go to [Google Colab](https://colab.research.google.com/)
- Click on **New Notebook**

### 2. Install Dependencies
Run the following command in a Colab code cell to install the required packages:

```python
!pip install gradio pandas plotly pydantic-ai
```

### 3. Clone the GitHub Repository
Replace `<your-repo-url>` with the actual GitHub repository URL:

```python
!git clone <your-repo-url>
%cd <repository-folder>
```

### 4. Run the Application
Execute the following command to launch the Gradio app:

```python
!python app.py
```

Once executed, Colab will generate a **Gradio link** where you can interact with the application.

## Uploading a CSV File
- When prompted in the Gradio UI, **upload a CSV file**.
- Ask questions about the dataset or generate visualizations.

## Troubleshooting
- If you face import errors, try reinstalling the dependencies:
  ```python
  !pip install --upgrade gradio pandas plotly pydantic-ai
  ```
- Ensure that the correct file paths are used in the script.

## Future Improvements
- Extend the application to support more advanced analytics.
- Add support for additional visualization types.

---
