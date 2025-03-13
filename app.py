import os
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Union
import re
from io import StringIO
import tempfile
import logging
import json
import requests
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a temporary directory to store uploaded files
temp_dir = tempfile.mkdtemp()
UPLOAD_FOLDER = temp_dir

# Install Ollama in Colab (this won't affect anything if you're not in Colab)
try:
    import google.colab
    # We're in Colab, install Ollama
    !curl -fsSL https://ollama.com/install.sh | sh
    !ollama pull llama3.1:8b-q4_0
except:
    # Not in Colab, continue normally
    pass

# Configure Ollama client - we'll use the REST API instead of a Python package
OLLAMA_BASE_URL = "http://localhost:11434/api"

class OllamaClient:
    """Simple client for Ollama API"""
    
    def __init__(self, base_url=OLLAMA_BASE_URL, model="llama3.1:8b-q4_0"):
        self.base_url = base_url
        self.model = model
    
    def generate(self, prompt, system_prompt=None):
        """Generate text from Ollama API"""
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        if system_prompt:
            data["system"] = system_prompt
            
        try:
            response = requests.post(f"{self.base_url}/generate", json=data)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return f"Error: {str(e)}"

# Initialize Ollama client
ollama = OllamaClient()

class DataFrameInfo:
    """Class to store and manage the uploaded DataFrame"""
    
    def __init__(self):
        self.df = None
        self.file_path = None
        self.column_info = None
        self.file_info = {}
    
    def load_df(self, file_path):
        """Load DataFrame from CSV file"""
        try:
            self.df = pd.read_csv(file_path)
            self.file_path = file_path
            self.update_column_info()
            self.update_file_info()
            return True
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            return False
    
    def update_column_info(self):
        """Update column information"""
        if self.df is not None:
            # Get column types and basic stats
            self.column_info = {}
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                sample = self.df[col].iloc[0:3].tolist()
                
                stats = {}
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    stats = {
                        "min": float(self.df[col].min()),
                        "max": float(self.df[col].max()),
                        "mean": float(self.df[col].mean()),
                        "median": float(self.df[col].median()),
                        "std": float(self.df[col].std()),
                        "null_count": int(self.df[col].isna().sum())
                    }
                else:
                    # For non-numeric columns
                    unique_values = self.df[col].nunique()
                    most_common = self.df[col].value_counts().nlargest(3).to_dict()
                    most_common = {str(k): int(v) for k, v in most_common.items()}
                    stats = {
                        "unique_values": int(unique_values),
                        "most_common": most_common,
                        "null_count": int(self.df[col].isna().sum())
                    }
                    
                self.column_info[col] = {
                    "dtype": dtype,
                    "sample": sample,
                    "stats": stats
                }
    
    def update_file_info(self):
        """Update file metadata"""
        if self.df is not None:
            self.file_info = {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "column_names": list(self.df.columns),
                "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "file_path": self.file_path,
                "file_name": os.path.basename(self.file_path) if self.file_path else None
            }

# Initialize DataFrame info storage
df_info = DataFrameInfo()

# Define Pydantic models for structured data
class AnswerResponse(BaseModel):
    """Structured response to a user's question"""
    answer: str
    requires_visualization: bool = False
    suggested_visualization: Optional[str] = None
    visualization_columns: List[str] = []
    visualization_type: Optional[str] = None
    code_to_execute: Optional[str] = None

class ChartDescription(BaseModel):
    """Describes a chart to be created"""
    chart_type: str
    x_column: str
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    title: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    additional_parameters: Dict[str, Any] = {}

def answer_question(question):
    """
    Use Ollama to analyze the question and dataset to provide a comprehensive answer.
    """
    if df_info.df is None:
        return AnswerResponse(answer="No dataset loaded. Please upload a CSV file first.")
    
    # Convert column info and file info to JSON strings for the prompt
    column_info_str = json.dumps(df_info.column_info, indent=2)
    file_info_str = json.dumps(df_info.file_info, indent=2)
    
    # Get sample data
    sample_data_str = df_info.df.head(5).to_string()
    
    system_prompt = """
    You are a data analysis assistant that helps users understand CSV data.
    Your job is to answer questions about the dataset accurately based on the dataset information provided.
    When appropriate, suggest visualizations that would help answer the question.
    
    Always respond in JSON format with these fields:
    {
        "answer": "Your detailed answer here",
        "requires_visualization": true/false,
        "visualization_type": "chart type if needed",
        "visualization_columns": ["column1", "column2"],
        "code_to_execute": "Python code using Plotly to create the visualization"
    }
    
    If you recommend a visualization, include Python code using Plotly that will create an appropriate chart.
    The code should create a variable called 'fig' containing the Plotly figure.
    The dataframe is available as the variable 'df'.
    """
    
    user_prompt = f"""
    Question: {question}
    
    Dataset Information:
    File Info: {file_info_str}
    
    Column Info: {column_info_str}
    
    Sample Data:
    {sample_data_str}
    
    Please analyze this question about the dataset and provide a JSON response with your answer.
    If a visualization would be helpful, include the appropriate Plotly code in the "code_to_execute" field.
    """
    
    # Get response from Ollama
    response_text = ollama.generate(user_prompt, system_prompt)
    
    # Extract JSON from response
    try:
        # Find JSON-like content in the response
        json_match = re.search(r'({[\s\S]*})', response_text)
        if json_match:
            json_str = json_match.group(1)
            response_dict = json.loads(json_str)
            
            # Convert to Pydantic model
            return AnswerResponse(**response_dict)
        else:
            # If no JSON found, create a basic response
            return AnswerResponse(
                answer=f"Received response but couldn't parse it as JSON. Raw answer: {response_text}"
            )
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return AnswerResponse(
            answer=f"Error processing response: {str(e)}. Raw response: {response_text}"
        )

def create_visualization_code(chart_request):
    """
    Use Ollama to generate code for creating the requested visualization using Plotly.
    """
    system_prompt = """
    You are a data visualization expert. 
    Your job is to create Python code using Plotly to visualize data based on the request.
    
    Your response should be ONLY the Python code, with no explanations or markdown.
    The code should create a variable called 'fig' containing the Plotly figure.
    The dataframe is available as the variable 'df'.
    """
    
    user_prompt = f"""
    Create Python code using Plotly that will generate a {chart_request.chart_type} chart.
    
    Use x_column: {chart_request.x_column}
    Use y_column: {chart_request.y_column if chart_request.y_column else "None"}
    Use color_column: {chart_request.color_column if chart_request.color_column else "None"}
    Chart title: {chart_request.title}
    
    The dataframe is called 'df'.
    Create a Plotly figure and assign it to a variable called 'fig'.
    Return only the Python code.
    """
    
    # Get response from Ollama
    return ollama.generate(user_prompt, system_prompt)

# Function to handle CSV upload
def upload_file(file):
    if file is None:
        return None, "No file uploaded.", None
    
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        
        # Load the DataFrame
        success = df_info.load_df(file_path)
        if not success:
            return None, "Failed to load CSV file. Please check the format.", None
        
        # Generate preview
        preview = df_info.df.head(5).to_html(classes='table table-striped')
        
        # Generate summary
        rows, cols = df_info.df.shape
        column_types = df_info.df.dtypes.to_dict()
        column_types = {k: str(v) for k, v in column_types.items()}
        
        summary = f"""
        ### Dataset Summary
        - File: {os.path.basename(file_path)}
        - Rows: {rows}
        - Columns: {cols}
        - Memory Usage: {df_info.file_info['memory_usage_mb']} MB
        
        ### Column Types
        """
        for col, dtype in column_types.items():
            summary += f"- {col}: {dtype}\n"
        
        return preview, summary, file_path
    
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return None, f"Error processing file: {str(e)}", None

# Function to process user questions
def process_question(question, file_path):
    if file_path is None or df_info.df is None:
        return "Please upload a CSV file first.", None
    
    if not question or question.strip() == "":
        return "Please enter a question.", None
    
    try:
        # Get response from LLM
        response = answer_question(question)
        
        fig = None
        if response.requires_visualization and response.code_to_execute:
            try:
                # Create a local environment with the dataframe
                local_env = {"df": df_info.df, "px": px, "go": go}
                
                # Execute the visualization code
                exec(response.code_to_execute, local_env)
                
                # Get the figure from the local environment
                fig = local_env.get('fig', None)
                
                # If no figure was created, add that to the response
                if fig is None:
                    response.answer += "\n\nNote: A visualization was suggested, but couldn't be created."
            except Exception as viz_error:
                logger.error(f"Visualization error: {str(viz_error)}")
                response.answer += f"\n\nNote: Tried to create a visualization but encountered an error: {str(viz_error)}"
        
        return response.answer, fig
    
    except Exception as e:
        logger.error(f"Error in process_question: {str(e)}")
        return f"Error processing question: {str(e)}", None

# Function to generate visualization based on user selections
def generate_visualization(chart_type, x_column, y_column, color_column, title, file_path):
    if file_path is None or df_info.df is None:
        return "Please upload a CSV file first.", None
    
    if not x_column:
        return "Please select at least X-axis column.", None
    
    try:
        # Create chart request
        chart_request = ChartDescription(
            chart_type=chart_type,
            x_column=x_column,
            y_column=y_column if y_column != "None" else None,
            color_column=color_column if color_column != "None" else None,
            title=title if title else f"{chart_type.capitalize()} of {x_column} vs {y_column}" 
        )
        
        # Get visualization code from LLM
        code_snippet = create_visualization_code(chart_request)
        
        try:
            # Create a local environment with the dataframe
            local_env = {"df": df_info.df, "px": px, "go": go}
            
            # Execute the visualization code
            exec(code_snippet, local_env)
            
            # Get the figure from the local environment
            fig = local_env.get('fig', None)
            
            if fig is None:
                return "Failed to create visualization. No figure was generated.", None
            
            return "Visualization generated successfully!", fig
            
        except Exception as viz_error:
            logger.error(f"Visualization execution error: {str(viz_error)}")
            return f"Error creating visualization: {str(viz_error)}\n\nCode attempted:\n{code_snippet}", None
    
    except Exception as e:
        logger.error(f"Error in generate_visualization: {str(e)}")
        return f"Error generating visualization: {str(e)}", None

# Create Gradio interface
with gr.Blocks(title="CSV Question Answering & Visualization") as app:
    gr.Markdown("# CSV Question Answering & Visualization")
    gr.Markdown("Upload a CSV file, ask questions about it, and visualize the data.")
    
    # File upload section
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV File")
            file_path = gr.State(None)
        
    with gr.Row():
        preview_output = gr.HTML(label="Data Preview")
        summary_output = gr.Markdown(label="Data Summary")
    
    # Question answering section
    gr.Markdown("## Ask Questions About Your Data")
    with gr.Row():
        question_input = gr.Textbox(label="Question", placeholder="Ask something about the data...")
        question_button = gr.Button("Ask")
    
    with gr.Row():
        answer_output = gr.Markdown(label="Answer")
        question_plot = gr.Plot(label="Visualization")
    
    # Visualization section
    gr.Markdown("## Custom Visualization")
    with gr.Row():
        chart_type = gr.Dropdown(
            label="Chart Type",
            choices=["scatter", "line", "bar", "histogram", "box", "pie", "heatmap"],
            value="scatter"
        )
        x_column = gr.Dropdown(label="X-Axis Column")
        y_column = gr.Dropdown(label="Y-Axis Column (Optional for some charts)")
        color_column = gr.Dropdown(label="Color/Group By Column (Optional)")
    
    with gr.Row():
        chart_title = gr.Textbox(label="Chart Title", placeholder="Enter chart title...")
        viz_button = gr.Button("Generate Visualization")
    
    custom_plot = gr.Plot(label="Custom Visualization")
    
    # Set up event handlers
    file_input.upload(
        fn=upload_file,
        inputs=[file_input],
        outputs=[preview_output, summary_output, file_path],
        api_name="upload_csv"
    ).then(
        fn=lambda file_path: (
            gr.Dropdown.update(choices=["None"] + list(df_info.df.columns) if df_info.df is not None else ["None"]),
            gr.Dropdown.update(choices=["None"] + list(df_info.df.columns) if df_info.df is not None else ["None"]),
            gr.Dropdown.update(choices=["None"] + list(df_info.df.columns) if df_info.df is not None else ["None"])
        ),
        inputs=[file_path],
        outputs=[x_column, y_column, color_column],
        api_name="update_columns"
    )
    
    question_button.click(
        fn=process_question,
        inputs=[question_input, file_path],
        outputs=[answer_output, question_plot],
        api_name="ask_question"
    )
    
    viz_button.click(
        fn=generate_visualization,
        inputs=[chart_type, x_column, y_column, color_column, chart_title, file_path],
        outputs=[answer_output, custom_plot],
        api_name="create_visualization"
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)