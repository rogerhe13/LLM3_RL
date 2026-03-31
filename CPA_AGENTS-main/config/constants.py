"""
Constants and configuration file 
"""

# ENHANCED: Query Understanding Agent Prompt - Handles all query types
QUERY_UNDERSTANDING_PROMPT = """
Analyze this user query about clinical performance: "{user_query}"

Extract ALL the following information and respond with a JSON object:

1. "query_type": Primary analysis type:
   - "temporal_trends" (changes/improvement over time)
   - "current_strengths" (what am I good at)
   - "areas_for_improvement" (what should I work on)
   - "specific_skill_analysis" (focused on one competency)
   - "rotation_specific" (performance in specific rotation(s))
   - "comparative_analysis" (comparing rotations/EPAs/time periods)
   - "pattern_recognition" (themes/patterns across feedback)
   - "general_performance" (overall how am I doing)

2. "competency_focus": Specific skill area (null if multiple/general):
   - "clinical_reasoning" (diagnostic thinking, decision-making, differential diagnosis, clinical judgment, problem-solving, assessment, planning)
   - "communication" (patient interaction, bedside manner, listening, shared decision-making)
   - "professionalism" (reliability, ethics, responsibility, feedback acceptance)
   - "patient_care" (empathy, compassion, patient advocacy)
   - "documentation" (note writing, charting)
   - "presentation_skills" (oral presentations, case presentations)
   - "teamwork" (collaboration, team contribution)
   - "history_taking" (H&P, physical exam skills)
   - "medical_knowledge" (fund of knowledge, literature use)

3. "temporal_dimension": true if asking about changes over time

4. "specific_numbers": Extract any numbers requested:
   - "strengths_requested": number or null
   - "improvements_requested": number or null
   - "top_requested": number or null (for "top 5 areas", etc.)

5. "rotation_filters": ONLY extract rotation names explicitly mentioned in the query:
   - ONLY if the rotation name actually appears in the user's query text
   - "What are my strengths?" → [] (no rotation mentioned)
   - "How did I do in [specific rotation]?" → [that rotation] (rotation explicitly mentioned)
   - Return empty list [] if no specific rotations mentioned

6. "epa_filters": List of specific EPAs mentioned:
   - Extract: ["epa1", "epa2", etc.] or ["EPA1", "EPA2", etc.]
   - Return empty list [] if no specific EPAs mentioned

7. "comparison_elements": If comparing things, what is being compared?
   - "rotations": [list of rotations] (if comparing rotations)
   - "time_periods": ["early", "recent"] (if comparing time periods)
   - "competencies": [list of skills] (if comparing skills)
   - null if no comparison requested

8. "evidence_criteria": What feedback text should be prioritized?
   - Be specific but inclusive for the competency focus
   - For clinical reasoning: include decision-making, presentations about reasoning, diagnostic thinking, assessment/planning
   - Don't be overly restrictive

Examples:
- "How has my clinical reasoning improved over time?" → temporal_trends, clinical_reasoning, temporal=true, rotation_filters=[]
- "What are my top 3 strengths in [rotation_name]?" → current_strengths, no focus, rotation_filters=[rotation_name], strengths_requested=3
- "Compare my [rotation_A] vs [rotation_B] performance" → comparative_analysis, no focus, comparison_elements={{"rotations": ["rotation_A", "rotation_B"]}}
- "How are my EPA1 and EPA2 scores?" → specific_skill_analysis, clinical_reasoning, epa_filters=["epa1", "epa2"]
- "What patterns do evaluators mention about my [skill_area]?" → pattern_recognition, [skill_area]

Return only valid JSON, no other text.
"""
#text analysis prompt 
TEXT_ANALYSIS_PROMPT = """
You are analyzing clinical performance feedback to answer this specific user query: "{user_query}"

Query Analysis:
- Query Type: {query_type}
- Competency Focus: {competency_focus}
- Temporal Dimension: {temporal_dimension}
- Rotation Filters: {rotation_filters}
- EPA Filters: {epa_filters}
- Numbers Requested: {specific_numbers}
- Evidence Criteria: {evidence_criteria}

Assessment Data: {parsed_data}

ENHANCED CLINICAL REASONING DETECTION:

For CLINICAL REASONING queries, ONLY include feedback that contains these SPECIFIC indicators:

1. **DIRECT CLINICAL REASONING TERMS** (High Priority - Must Include):
   - "clinical reasoning", "clinical judgment", "clinical thinking"
   - "diagnostic reasoning", "diagnostic thinking", "diagnostic approach"
   - "differential diagnosis", "DDx", "differential"
   - "decision-making", "medical decision", "clinical decision"
   - "problem-solving", "analytical thinking", "clinical analysis"
   - "assessment and plan", "A&P", "clinical planning"
   - "synthesizing information", "integrating findings"
   - "evidence-based", "literature application", "applying knowledge"

2. **REASONING THROUGH CLINICAL ACTIONS** (Medium Priority):
   - "thought process", "reasoning process", "explains reasoning"
   - "clinical approach", "systematic approach", "methodical"
   - "connects findings", "links symptoms", "correlates data"
   - "anticipates", "predicts", "foresees clinical needs"
   - "complex cases", "complicated patients", "challenging diagnosis"

3. **PRESENTATION SKILLS SHOWING REASONING** (Include Only If Reasoning-Focused):
   - "presents differential", "discusses reasoning", "explains thought process"
   - "organized thinking", "logical progression", "systematic presentation"
   - "shows clinical thinking", "demonstrates reasoning"
   
   BUT EXCLUDE general presentation comments like:
   - "presents well", "good presentations", "concise presentations"
   - Unless they specifically mention reasoning/thinking process

4. **STRICT EXCLUSIONS** (Never Include These for Clinical Reasoning):
   - General work traits: "hard working", "thorough", "reliable", "punctual"
   - Attitude: "great attitude", "positive", "enthusiastic", "team player" 
   - Basic skills: "asks questions", "prepared", "organized", "professional"
   - Reading habits: "reads about patients", "studies", "looks things up"
   - Communication: "good bedside manner", "communicates well"
   - Unless these terms are specifically linked to reasoning (e.g., "asks thoughtful diagnostic questions")

5. **RELEVANCE SCORING**:
   - HIGH: Contains direct clinical reasoning terms from list 1
   - MEDIUM: Contains reasoning-related actions from list 2, but no general traits
   - LOW: General positive feedback without specific reasoning content
   - EXCLUDE: Only contains terms from exclusion list

6. **TEMPORAL ANALYSIS ENHANCEMENT**:
   If temporal_dimension=true:
   - ONLY track progression in actual reasoning-related feedback
   - Compare early vs recent REASONING-SPECIFIC comments
   - If no reasoning-specific feedback exists, state: "Limited specific clinical reasoning feedback available for temporal analysis"

Example of GOOD clinical reasoning feedback:
- "Demonstrates excellent clinical reasoning when working through complex cases"
- "Shows strong diagnostic thinking and develops appropriate differential diagnoses"
- "Clinical decision-making has improved significantly"

Example of EXCLUDED general feedback:
- "Very thorough and hard working" ❌ (General work trait)
- "Asks appropriate questions and presents really well" ❌ (Unless questions are specified as diagnostic)
- "Continue to read on your patients" ❌ (Study habit, not reasoning)

CRITICAL RULE: If no HIGH or MEDIUM relevance feedback is found for clinical reasoning, return:
- relevant_feedback_found: false
- alternative_suggestions: "Available feedback focuses on [other areas like work habits, communication, etc.]. For clinical reasoning assessment, look for feedback on diagnostic thinking, clinical judgment, or decision-making processes."

Return JSON format:
{{
  "relevant_feedback_found": true/false,
  "clinical_reasoning_specific": {{
    "high_relevance_feedback": [
      {{
        "text": "Original quote",
        "reasoning_terms_found": ["specific terms that made this relevant"],
        "evaluator_role": "role",
        "rotation": "rotation name", 
        "date": "date",
        "relevance_explanation": "Why this specifically relates to clinical reasoning"
      }}
    ],
    "medium_relevance_feedback": [...],
    "excluded_general_feedback": [
      {{
        "text": "Excluded quote",
        "exclusion_reason": "Why this was excluded (general work trait/attitude/etc.)"
      }}
    ]
  }},
  "competency_analysis": {{
    "strengths": [...],
    "improvements": [...],
    "temporal_progression": {{
      "reasoning_specific_trend": "improving/declining/stable/insufficient_data",
      "early_reasoning_feedback": [...],
      "recent_reasoning_feedback": [...],
      "note": "If insufficient data, explain what type of feedback IS available"
    }}
  }},
  "alternative_suggestions": "If limited clinical reasoning feedback, suggest other rich areas"
}}

REMEMBER: Be extremely strict about clinical reasoning relevance. Better to say "insufficient specific feedback" than to include general work performance comments.
"""

# ENHANCED: Numeric Analysis Agent Prompt - Better temporal and comparative analysis
NUMERIC_ANALYSIS_PROMPT = """
Analyze clinical performance numerical data for this query: "{user_query}"

Query Context:
- Query Type: {query_type}
- Competency Focus: {competency_focus}
- Temporal Analysis Needed: {temporal_dimension}
- Rotation Filters: {rotation_filters}
- EPA Filters: {epa_filters}

Parsed Data: {parsed_data}

ANALYSIS REQUIREMENTS:

1. BASIC STATISTICS: Calculate for all relevant scores:
   - Weighted averages (using recency_weight)
   - Min/max values
   - Number of evaluations
   - Score distributions

2. TEMPORAL ANALYSIS (if temporal_dimension=true):
   - Sort all scores chronologically by release_date
   - Calculate trends for each EPA/competency area
   - Compare early period vs recent period averages
   - Identify improving/declining/stable patterns
   - Include specific date ranges and score changes

3. ROTATION-SPECIFIC ANALYSIS (if rotation_filters provided):
   - Calculate statistics only for specified rotations
   - Compare performance across different rotations
   - Identify rotation-specific patterns

4. COMPETENCY-FOCUSED ANALYSIS:
   - For clinical_reasoning: Focus on EPA1, EPA2, EPA3, EPA7
   - For communication: Focus on comm_* fields and EPA6
   - For professionalism: Focus on prof_* fields
   - For patient_care: Focus on EPA1, EPA9, professionalism scores

5. COMPARATIVE ANALYSIS:
   - If comparing rotations: provide side-by-side statistics
   - If comparing time periods: show before/after metrics
   - If comparing competencies: rank different skill areas

Return JSON format:
{{
  "query_specific_analysis": {{
    "primary_metrics": {{"metric_name": {{"current": value, "trend": "direction", "change": numeric_change}}}},
    "temporal_progression": {{
      "time_span": "date range",
      "early_period": {{"avg_scores": {{}}, "date_range": ""}},
      "recent_period": {{"avg_scores": {{}}, "date_range": ""}},
      "trends": {{"epa_name": {{"direction": "improving/stable/declining", "magnitude": 0.0-1.0}}}}
    }},
    "rotation_comparison": {{
      "rotation_name": {{"avg_scores": {{}}, "evaluation_count": number}}
    }}
  }},
  "by_epa": {{
    "epa_name": {{"avg": value, "trend": {{}}, "rotation_breakdown": {{}}}}
  }},
  "by_competency": {{
    "clinical_reasoning": {{"composite_score": value, "trend": {{}}}},
    "communication": {{"composite_score": value, "trend": {{}}}},
    "professionalism": {{"composite_score": value, "trend": {{}}}}
  }},
  "summary_insights": [
    "Key quantitative insight 1",
    "Key quantitative insight 2"
  ]
}}

Focus on metrics most relevant to their specific query. Provide detailed temporal analysis if requested.
"""

# ENHANCED: Consolidation Agent Prompt - Handles all query types
CONSOLIDATION_PROMPT = """
You are consolidating clinical performance analysis results for this query: "{user_query}"

Query Analysis: {structured_query}
Numeric Analysis: {numeric_analysis}  
Text Analysis: {text_analysis}

QUERY TYPE SPECIFIC CONSOLIDATION:

1. For TEMPORAL_TRENDS queries:
   - Focus on progression over time with specific dates
   - Compare early vs recent feedback with quotes
   - Include EPA score trends with numerical changes
   - Highlight what specifically changed

2. For CURRENT_STRENGTHS queries:
   - Rank strengths by frequency and confidence
   - Provide specific evidence from multiple evaluators
   - Focus on consistent patterns across rotations
   - If number requested (e.g., "top 3"), prioritize accordingly

3. For AREAS_FOR_IMPROVEMENT queries:
   - Identify actionable development areas
   - Group similar feedback themes
   - Provide context on how often mentioned
   - Suggest specific next steps

4. For ROTATION_SPECIFIC queries:
   - Focus only on the requested rotation(s)
   - Compare performance across different rotations if multiple
   - Highlight rotation-specific strengths/challenges
   - Include rotation-specific EPA scores

5. For COMPARATIVE_ANALYSIS queries:
   - Direct comparison between requested elements
   - Quantify differences where possible
   - Highlight unique aspects of each comparison target
   - Use specific evidence to support comparisons

6. For PATTERN_RECOGNITION queries:
   - Identify recurring themes across evaluations
   - Note consistency of feedback across evaluators/rotations
   - Highlight evolution of patterns over time
   - Group related feedback together

Return JSON format:
{{
  "summary": "Brief summary directly answering the user's specific question",
  "key_findings": [
    {{
      "category": "strength/improvement/trend/pattern/comparison",
      "title": "Specific finding title",
      "description": "What this means for the student",
      "evidence": ["Direct quotes or specific data points"],
      "confidence": "high/medium/low",
      "source_count": "X evaluators across Y rotations",
      "temporal_context": "When this was observed (if relevant)"
    }}
  ],
  "numeric_context": {{
    "relevant_scores": {{"score_name": value}},
    "trends": "Quantitative trend information",
    "comparisons": "Numerical comparisons if relevant"
  }},
  "rotation_breakdown": {{
    "rotation_name": {{
      "key_points": ["rotation-specific insights"],
      "scores": {{"epa_name": value}}
    }}
  }},
  "temporal_analysis": {{
    "time_span": "Date range analyzed",
    "progression": "How performance changed over time",
    "early_vs_recent": "Specific comparison with dates"
  }},
  "data_quality": {{
    "total_evaluations": "number",
    "evaluator_types": ["Attending", "Resident", etc.],
    "rotations_covered": ["Surgery", "Medicine", etc.],
    "confidence_assessment": "How reliable this analysis is"
  }}
}}

Focus specifically on what the user asked about. Don't provide generic summaries unless requested.
Return only valid JSON.
"""

# ENHANCED: Response Generation Prompt - Tailored responses for each query type
# MERGED: Response Generation Prompt - Enhanced intelligence + Old detailed formatting
RESPONSE_GENERATION_PROMPT = """
You are helping a medical student understand their clinical performance assessment data.

Original user query: {user_query}
Structured query analysis: {structured_query}
Consolidated summary: {consolidated_summary}
Pattern information: {pattern_info}
Raw evidence data: {raw_evidence}

CRITICAL GUIDELINES:

1. ANSWER THE ACTUAL QUESTION: Focus specifically on what the user asked about.
   - If they asked about clinical reasoning over time, focus on that progression
   - If they asked for 3 strengths, provide exactly 3
   - Don't provide generic performance summary unless requested

2. RELEVANCE CHECK: Only use feedback that actually relates to the query.
   - If consolidated summary shows no relevant feedback, say so honestly
   - Don't force unrelated feedback into your answer
   - Use the "alternative_suggestions" if no direct match found

3. TEMPORAL ANALYSIS: For "over time" queries:
   - Clearly explain progression patterns
   - Compare early vs recent feedback with specific dates
   - Use trend analysis from numeric data
   - Be specific about improvement/decline

4. EVIDENCE QUALITY: Prioritize high-relevance feedback:
   - Use quotes marked as "high relevance" first
   - Note confidence levels from pattern analysis
   - Explain why certain patterns are reliable (multiple evaluators, etc.)

5. MANDATORY FORMAT (for each main point):

**X. [Title - specific to what they asked about]**

**Analysis:** [Your interpretation in second-person - "You demonstrate..." "Your performance shows..."]

**Supporting Evidence:**
- "[Keep original quotes exactly as written - do NOT change pronouns in quotes]" - [Evaluator Role] ([Rotation], [Date])
- "[Keep original quotes exactly as written - do NOT change pronouns in quotes]" - [Evaluator Role] ([Rotation], [Date])
- "[Keep original quotes exactly as written - do NOT change pronouns in quotes]" - [Evaluator Role] ([Rotation], [Date])

**Pattern Confidence:** [Level] ([Number] evaluators across [rotations/timeframe])

**Related Performance Data:** [EPA scores, trends, etc.]

6. RESPONSE FORMAT BY QUERY TYPE:

TEMPORAL QUERIES ("over time", "improved", "changed"):
Format: 
# Clinical Reasoning Progression Over Time

**Overall Trend:** [Clear statement of progression]

**Early Performance** (Date Range):
[Specific examples from early evaluations with quotes and dates using the mandatory format above]

**Recent Performance** (Date Range):
[Specific examples from recent evaluations with quotes and dates using the mandatory format above]

**Key Changes Observed:**
- [Specific change 1 with evidence]
- [Specific change 2 with evidence]

**Quantitative Trends:** [EPA scores or other metrics over time]

STRENGTHS/IMPROVEMENTS QUERIES ("top 3 strengths", "areas to improve"):
Format:
# Your [Top X] [Strengths/Areas for Improvement]

[Use the mandatory format above for each item, numbered 1, 2, 3, etc.]

ROTATION-SPECIFIC QUERIES ("Surgery performance", "Medicine vs Pediatrics"):
Format:
# Performance in [Rotation Name]

**Overall Assessment:** [Summary for this rotation]

**Key Strengths in [Rotation]:**
[Use mandatory format above for rotation-specific strengths]

**Development Areas in [Rotation]:**
[Use mandatory format above for rotation-specific improvements]

**Rotation-Specific Scores:** [Relevant EPA scores]

COMPETENCY-SPECIFIC QUERIES ("communication skills", "clinical reasoning"):
Format:
# [Competency Name] Analysis

**Current Performance:** [Overall assessment]

**Demonstrated Strengths:**
[Use mandatory format above for competency-specific strengths]

**Development Opportunities:**
[Use mandatory format above for areas for growth]

7. QUOTE PRESERVATION: 
   - Keep all quoted evidence exactly as written in the original feedback
   - Do NOT change "he/she" to "you" in direct quotes
   - Only use second-person in your analysis sections
   - Example: 
     Analysis: "You demonstrate strong reasoning..." 
     Evidence: "He shows excellent judgment" - Attending

8. EVIDENCE REQUIREMENTS:
   - Provide at least 3 pieces of supporting evidence for each major point
   - Use evidence from different evaluators when possible
   - Use evidence from different rotations when possible
   - Include evaluator role, rotation, and date for each quote

9. HONEST COMMUNICATION:
   - If insufficient data: "Based on available feedback, there isn't enough specific information about [topic]. However, I can share insights about [related areas]..."
   - If no temporal data: "Your feedback doesn't show clear progression over time, but here's what evaluators consistently note..."

10. ALWAYS use second-person perspective in analysis ("You demonstrate..." not "The student demonstrates...")

The tone should be professional, constructive, and supportive - appropriate for medical education context.

Return a well-formatted response that directly answers the user's question with authentic quoted evidence using the mandatory format above.
"""

# Keep existing working prompts unchanged
DATA_INGESTION_PROMPT = """
Your task is to convert CSV table data into a standardized JSON format.
Please use the provided column type mapping to convert each field to the appropriate type.

Column names: {columns}
Column type mapping: {column_type_map}
Raw data: {raw_table}

Please convert the data to a JSON array format, such as:
[
  {{
    "student_id": "001dca3071ef3fa402c8463bcc0bb952",
    "epa1": 3,
    "strengths_comment": "- presentations are concise, organized, well thought out..."
  }},
  ...
]

Ensure all numeric fields are converted to number types, date fields are properly formatted, and text fields remain unchanged.
"""

ORCHESTRATOR_PROMPT = """
You are the orchestrator agent of the clinical performance data analysis system.
Your task is to coordinate the work of all other agents and ensure data flows correctly from one agent to the next.

You need to follow these steps:
1. Receive the user query and raw table data
2. Call the data ingestion agent to process the table data
3. Call the query understanding agent to analyze the user query
4. Call the numeric analysis agent to calculate statistics
5. Call the text analysis agent to extract text feedback
6. Call the consolidation agent to merge analysis results
7. Call the response generation agent to create the final answer

User query: {user_query}
Raw table: {raw_table_summary}

Please coordinate the work of each agent according to the steps above, and return the final natural language response.
"""