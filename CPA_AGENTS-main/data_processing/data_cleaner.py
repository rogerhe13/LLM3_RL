# data_processing/data_cleaner.py
import pandas as pd
import os
import re
from typing import Dict, List, Any, Optional

class DataCleaner:
   """Data cleaner for medical student assessment data"""
   
   def clean_data(self, input_file_path: str, output_file_path: str = None) -> str:
       """
       Clean assessment data and save to specified location
       
       Args:
           input_file_path: Path to raw data file
           output_file_path: Path where clean data should be saved
                           If None, saves to cpa_data/cpa_clean.csv
       
       Returns:
           Path to cleaned file
       """
       if output_file_path is None:
           output_file_path = "cpa_data/cpa_clean.csv"
       
       # Ensure output directory exists
       os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
       
       try:
           # Read and process data
           df = pd.read_csv(input_file_path, dtype=str)
           df = df.fillna('')
           
           clean_data = self._process_data(df)
           
           # Save cleaned data
           clean_data.to_csv(output_file_path, index=False)
           
           return output_file_path
           
       except Exception as e:
           raise Exception(f"Error cleaning data: {str(e)}")
   
   def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Process and clean the assessment data
       
       Args:
           df: Input DataFrame with raw data
           
       Returns:
           DataFrame with cleaned data
       """
       # Create a list to store clean rows
       clean_rows = []
       
       # Group by student
       student_groups = df.groupby('student')
       
       for student_id, student_df in student_groups:
           # Skip empty student IDs
           if not student_id or pd.isna(student_id):
               continue
               
           # Group by form submission (form, phase, year, date)
           # Create a temporary column for grouping
           student_df = student_df.copy()
           student_df['form_key'] = student_df.apply(
               lambda row: f"{row.get('formname', '')}|{row.get('phasename', '')}|{row.get('academicyearname', '')}|{row.get('releasedate', '')}", 
               axis=1
           )
           
           # Group by form key
           form_groups = student_df.groupby('form_key')
           
           for form_key, form_df in form_groups:
               # Skip invalid form keys
               if '||' in form_key:
                   continue
                   
               # Extract form metadata
               form_parts = form_key.split('|')
               if len(form_parts) != 4:
                   continue
                   
               form_name, phase_name, academic_year, release_date = form_parts
               
               # Find evaluator blocks
               evaluator_blocks = self._identify_evaluator_blocks(form_df)
               
               # Process each evaluator block
               for evaluator_df in evaluator_blocks:
                   # Skip blocks without role information
                   role_row = evaluator_df[evaluator_df['questionname'] == "Please select your role:"]
                   if role_row.empty:
                       continue
                       
                   # Extract role and frequency
                   role = role_row['questionchoicetext'].iloc[0] if not role_row['questionchoicetext'].empty else ''
                   
                   frequency_row = evaluator_df[evaluator_df['questionname'] == "Frequency"]
                   frequency = frequency_row['questionchoicetext'].iloc[0] if not frequency_row.empty and not frequency_row['questionchoicetext'].empty else ''
                   
                   # Extract professionalism ratings
                   prof_rows = evaluator_df[(evaluator_df['questionname'] == "Professionalism:") & 
                                         (evaluator_df['rating_answer_sortorder'].notna())]
                   
                   # Extract communication ratings - include both Communication: questions and CES competency
                   comm_rows = evaluator_df[
                       (
                           evaluator_df['questionname'].str.contains("Communication:", na=False) |
                           evaluator_df['questionname'].str.contains("Advocates for patients by addressing social determinants", na=False) |
                           evaluator_df['questionname'].str.contains("CES competency", na=False)
                       ) & 
                       (evaluator_df['rating_answer_sortorder'].notna())
                   ]
                   
                   # Extract EPA skills
                   epa_rows = evaluator_df[evaluator_df['questionname'].str.contains("EPA", na=False) & 
                                        (evaluator_df['rating_answer_sortorder'].notna())]
                   
                   # Extract comments
                   strength_row = evaluator_df[evaluator_df['text_answer_category'] == "positive"]
                   strength_comment = self._clean_comment(strength_row['text_answer'].iloc[0]) if not strength_row.empty else ''
                   
                   improvement_row = evaluator_df[evaluator_df['text_answer_category'] == "improvement"]
                   improvement_comment = self._clean_comment(improvement_row['text_answer'].iloc[0]) if not improvement_row.empty else ''
                   
                   # Create base row with student and form info
                   base_row = {
                       'student_id': student_id,
                       'form_name': form_name,
                       'phase_name': phase_name,
                       'academic_year': academic_year,
                       'release_date': self._format_date(release_date),
                       'evaluator_role': role,
                       'frequency': frequency,
                       'strengths_comment': strength_comment,
                       'improvements_comment': improvement_comment
                   }
                   
                   # Add professionalism ratings
                   for _, row in prof_rows.iterrows():
                       question = row.get('ratingscalequestiontext', '')
                       if question:
                           key = f"prof_{self._convert_to_key(question)}"
                           base_row[key] = self._safe_int(row.get('rating_answer_sortorder'))
                   
                   # Add communication ratings
                   for _, row in comm_rows.iterrows():
                       question_name = row.get('questionname', '')
                       if 'Listening' in question_name or 'listening' in question_name:
                           key = "comm_listening"
                       elif 'shared decision' in question_name or 'decision making' in question_name:
                           key = "comm_decision_making"
                       elif 'Advocates for patients' in question_name or 'social determinants' in question_name or 'CES competency' in question_name:
                           key = "comm_advocacy"
                       else:
                           continue  # Skip unknown communication questions
                       
                       base_row[key] = self._safe_int(row.get('rating_answer_sortorder'))
                   
                   # Add EPA ratings
                   for _, row in epa_rows.iterrows():
                       question_name = row.get('questionname', '')
                       epa_match = re.search(r'EPA\s*(\d+)', question_name)
                       if epa_match:
                           epa_num = epa_match.group(1)
                           key = f"epa{epa_num}"
                           base_row[key] = self._safe_int(row.get('rating_answer_sortorder'))
                   
                   # Add row to results
                   clean_rows.append(base_row)
       
       # Create DataFrame from rows
       if not clean_rows:
           return pd.DataFrame()
       
       # Convert list of dictionaries to DataFrame
       result_df = pd.DataFrame(clean_rows)
       
       # Ensure consistent columns even if some data is missing
       # Add common columns that should always be present
       required_columns = [
           'student_id', 'form_name', 'phase_name', 'academic_year', 'release_date', 
           'evaluator_role', 'frequency', 'strengths_comment', 'improvements_comment'
       ]
       
       # Add professionalism columns
       prof_patterns = [
           'shows_dependability_truthfulness_and_integrity',
           'acknowledges_and_demonstrates_awareness_of_limitations',
           'takes_initiative_for_own_learning_and_patient_care',
           'remains_open_to_feedback_and_attempts_to_implement_it',
           'treats_all_patients_with_respect_and_compassion_protects_patient_confidentiality'
       ]
       prof_columns = [f'prof_{p}' for p in prof_patterns]
       
       # Add communication columns
       comm_columns = ['comm_listening', 'comm_decision_making', 'comm_advocacy', 'comm_other']
       
       # Add EPA columns (EPA 1-9 are common)
       epa_columns = [f'epa{i}' for i in range(1, 10)]
       
       # Combine all columns
       all_columns = required_columns + prof_columns + comm_columns + epa_columns
       
       # Add missing columns with NaN
       for col in all_columns:
           if col not in result_df.columns:
               result_df[col] = None
       
       return result_df

   def _identify_evaluator_blocks(self, form_df: pd.DataFrame) -> List[pd.DataFrame]:
       """
       Identify evaluator blocks in form data
       
       Args:
           form_df: DataFrame with form data
           
       Returns:
           List of DataFrames, each containing data for one evaluator
       """
       # Reset index to get position information
       form_df = form_df.reset_index(drop=True)
       
       # Find rows that start new evaluator blocks
       block_starts = form_df[form_df['questionname'] == "Please select your role:"].index.tolist()
       
       # Add end marker
       block_starts.append(len(form_df))
       
       # Extract blocks
       blocks = []
       for i in range(len(block_starts) - 1):
           start_idx = block_starts[i]
           end_idx = block_starts[i + 1]
           blocks.append(form_df.iloc[start_idx:end_idx].copy())
       
       return blocks

   def _convert_to_key(self, text: str) -> str:
       """Convert a question text to a snake_case key"""
       if not text:
           return ''
       
       # Remove non-alphanumeric characters, convert to lowercase
       text = re.sub(r'[^\w\s]', '', text.lower())
       
       # Replace spaces with underscores
       return re.sub(r'\s+', '_', text.strip())

   def _format_date(self, date_string: str) -> str:
       """Format date to ISO standard"""
       if not date_string:
           return ''
       
       try:
           # Parse the date string
           if ' ' in date_string:
               date_part, time_part = date_string.split(' ')
               month, day, year_short = date_part.split('/')
               year = f'20{year_short}'
               return f'{year}-{month.zfill(2)}-{day.zfill(2)}'
           else:
               return date_string  # Return as is if not in expected format
       except Exception:
           return date_string  # Return original string if parsing fails

   def _clean_comment(self, comment: str) -> str:
       """Clean comment text by removing placeholder tags"""
       if not comment or pd.isna(comment):
           return ''
       
       # Remove placeholder tags like <LOCATION>, <ADDRESSES>, etc.
       comment = re.sub(r'<[A-Z_]+>', '[REDACTED]', comment)
       # Fix common formatting issues
       comment = re.sub(r'\s+', ' ', comment).strip()
       
       return comment

   def _safe_int(self, value: Any) -> Optional[int]:
       """Safely convert a value to integer"""
       if value is None or pd.isna(value) or value == '':
           return None
       try:
           return int(value)
       except (ValueError, TypeError):
           return None
       
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    raw_data_path = os.getenv('RAW_DATA_PATH')
    
    print(f"RAW_DATA_PATH from .env: '{raw_data_path}'")  # Debug line
    
    if not raw_data_path:
        print("Please set RAW_DATA_PATH in .env file")
        exit(1)
    
    if not os.path.exists(raw_data_path):
        print(f"File not found: {raw_data_path}")
        exit(1)
    
    print(f"Processing: {raw_data_path}")
    cleaner = DataCleaner()
    cleaned_path = cleaner.clean_data(raw_data_path)
    print(f"Data cleaned and saved to: {cleaned_path}")