"""
Data processing helper functions
"""
from typing import List, Dict, Any, Union, Tuple
import re
import json
from datetime import datetime, timedelta

def clean_csv_data(raw_data: List[List[str]], columns: List[str]) -> List[Dict[str, Any]]:
    """
    Clean and convert raw CSV data
    
    Args:
        raw_data: Raw CSV row data
        columns: List of column names
        
    Returns:
        List of cleaned data dictionaries
    """
    cleaned_data = []
    
    for row in raw_data:
        
        if len(row) != len(columns):
            continue
            
        row_dict = {}
        for i, col_name in enumerate(columns):
            
            value = clean_cell_value(row[i], col_name)
            row_dict[col_name] = value
            
        # Add recency weight based on release date
        if 'release_date' in row_dict:
            row_dict['recency_weight'] = calculate_recency_weight(row_dict['release_date'])
            
            # Preserve original date string for display in evidence
            original_date = None
            for i, col_name in enumerate(columns):
                if col_name == 'release_date' and i < len(row):
                    original_date = row[i]
                    break
                    
            if original_date:
                row_dict['release_date_str'] = original_date
            
        cleaned_data.append(row_dict)
    
    return cleaned_data

def clean_cell_value(value: str, col_name: str) -> Any:
    """
    Clean and convert a single cell value
    
    Args:
        value: Raw cell value
        col_name: Column name
        
    Returns:
        Cleaned and converted value
    """
    
    if not value or str(value).lower() in ('none', 'null', 'na', 'n/a', '#name?'):
        return None
        
    
    if re.match(r'^epa\d+$', col_name) or col_name.startswith('prof_') or col_name.startswith('comm_'):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    
    if col_name == 'release_date':
        # Try multiple date formats
        for fmt in ('%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y'):
            try:
                return datetime.strptime(value, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format works, return the original value
        return value
    
    # Default keep as string
    return value

def parse_date(date_str: str) -> Union[datetime, None]:
    """
    Parse date string to datetime object
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Datetime object or None if parsing fails
    """
    if not date_str:
        return None
        
    # If already a datetime object
    if isinstance(date_str, datetime):
        return date_str
    
    # Try various formats including the standardized Y-m-d format
    for fmt in ('%Y-%m-%d', '%m/%d/%y', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
            
    return None

def calculate_recency_weight(date_value: Any) -> float:
    """
    Calculate recency weight based on date
    - 1.0 for dates within last 3 months
    - Linear decay from 3-9 months
    - 0.0 for dates older than 9 months
    
    Args:
        date_value: Date value (string, datetime, or ISO format)
        
    Returns:
        Recency weight between 0.0 and 1.0
    """
    if not date_value:
        return 0.5  # Default mid-weight for missing dates
        
    # Parse the date
    date_obj = parse_date(date_value)
    if not date_obj:
        return 0.5  # Default mid-weight if parsing fails
    
    # Calculate months between date and current date
    current_date = datetime.now()
    months_difference = (current_date.year - date_obj.year) * 12 + (current_date.month - date_obj.month)
    
    # Apply weighting rule
    if months_difference <= 3:
        # Full weight for last 3 months
        return 1.0
    elif months_difference >= 9:
        # Zero weight after 9 months
        return 0.0
    else:
        # Linear decay between 3-9 months
        return 1.0 - (months_difference - 3) / 6

def format_date_for_display(date_str: str) -> str:
    """
    Format date string for display in evidence citations
    
    Args:
        date_str: Date string in any supported format
        
    Returns:
        Formatted date string (e.g., "March 2023")
    """
    date_obj = parse_date(date_str)
    
    if date_obj:
        return date_obj.strftime("%B %Y")
    
    return date_str or "Unknown date"

def calculate_trend(scores_with_dates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate performance trend over time
    
    Args:
        scores_with_dates: List of dictionaries with date and score
        
    Returns:
        Dictionary with trend information
    """
    if not scores_with_dates or len(scores_with_dates) < 2:
        return {"direction": "stable", "magnitude": 0, "change": 0}
    
    # Convert dates and sort chronologically
    dated_scores = []
    for item in scores_with_dates:
        date_str = item.get("date")
        score = item.get("score")
        
        if date_str and score is not None:
            date_obj = parse_date(date_str)
            if date_obj:
                dated_scores.append({"date": date_obj, "score": float(score)})
    
    if len(dated_scores) < 2:
        return {"direction": "stable", "magnitude": 0, "change": 0}
    
    # Sort by date
    dated_scores.sort(key=lambda x: x["date"])
    
    # Get earliest and most recent scores
    earliest = dated_scores[0]["score"]
    most_recent = dated_scores[-1]["score"]
    
    # Calculate change
    change = most_recent - earliest
    
    # Determine trend direction and magnitude
    if abs(change) < 0.5:  # Less than half a point change
        direction = "stable"
    else:
        direction = "improving" if change > 0 else "declining"
    
    # Normalize magnitude to a 0-1 scale (assuming scores are on a 4-point scale)
    magnitude = min(1.0, abs(change) / 4.0)
    
    return {
        "direction": direction,
        "magnitude": round(magnitude, 2),
        "change": round(change, 2)
    }

def categorize_text(text: str, keyword_map: Dict[str, Dict[str, str]]) -> str:
    """
    Categorize text into a specific domain based on keywords
    
    Args:
        text: Text to categorize
        keyword_map: Keyword to domain mapping
        
    Returns:
        Matching domain name, or default domain if no match
    """
    default_domain = "General Performance"
    
    if not text:
        return default_domain
        
    text_lower = text.lower()
    
    # Check if each keyword exists in the text
    for keyword, mapping in keyword_map.items():
        if keyword.lower() in text_lower:
            return mapping.get("domain", default_domain)
    
    return default_domain

def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text
    
    Args:
        text: Original text
        
    Returns:
        List of sentences
    """
    if not text:
        return []
        
    
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def extract_bullet_points(text: str) -> List[str]:
    """
    Extract bullet points from text
    
    Args:
        text: Original text
        
    Returns:
        List of bullet points
    """
    if not text:
        return []
    
   
    bullet_points = re.split(r'[-•*]\s*', text)
    bullet_points = [point.strip() for point in bullet_points if point.strip()]
    
    
    if len(bullet_points) <= 1:
        return extract_sentences(text)
    
    return bullet_points

def limit_items(items: List[Any], count: int = None) -> List[Any]:
    """
    Limit number of items in a list based on relevance/importance
    
    Args:
        items: Original list
        count: Maximum number of items, if None return all items
        
    Returns:
        Limited list
    """
    if count is None or count >= len(items):
        return items
    
    # For scored items (dicts with a 'score' key), sort by score
    if items and isinstance(items[0], dict) and 'score' in items[0]:
        sorted_items = sorted(items, key=lambda x: x.get('score', 0), reverse=True)
        return sorted_items[:count]
    
    # For regular items, just take first N
    return items[:count]

def weighted_mean(values: List[float], weights: List[float] = None) -> float:
    """
    Calculate weighted mean
    
    Args:
        values: List of values
        weights: List of weights, if None equal weights are used
        
    Returns:
        Weighted mean
    """
    if not values:
        return 0.0
        
    if not weights:
        return sum(values) / len(values)
        
    if len(values) != len(weights):
        # Fallback to simple average if lengths don't match
        return sum(values) / len(values)
        
    numerator = sum(value * weight for value, weight in zip(values, weights))
    denominator = sum(weights)
    
    if denominator == 0:
        return 0.0
        
    return numerator / denominator