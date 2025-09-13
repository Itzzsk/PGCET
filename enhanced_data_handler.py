# Create enhanced_data_handler.py
import json
import pandas as pd
import numpy as np

class EnhancedPGCETDataHandler:
    def __init__(self, json_file='combined_pgcet_data.json'):
        self.data_file = json_file
        self.colleges_data = self.load_data()
        
    def load_data(self):
        """Load combined college data"""
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        print(f"📚 Loaded {len(data)} colleges from {self.data_file}")
        return data
    
    def search_by_rank_advanced(self, student_rank, category, round_preference=None):
        """Advanced rank-based search with multiple rounds"""
        eligible_colleges = []
        
        for college in self.colleges_data:
            # Check primary cutoffs
            cutoffs = college.get('cutoffs', {})
            rounds_data = college.get('rounds', {})
            
            # Determine which cutoff to use
            cutoff_rank = None
            round_used = 'Primary'
            
            if category in cutoffs and cutoffs[category] is not None:
                cutoff_rank = cutoffs[category]
            elif round_preference and round_preference in rounds_data:
                if category in rounds_data[round_preference] and rounds_data[round_preference][category] is not None:
                    cutoff_rank = rounds_data[round_preference][category]
                    round_used = round_preference
            else:
                # Try all available rounds
                for round_name, round_cutoffs in rounds_data.items():
                    if category in round_cutoffs and round_cutoffs[category] is not None:
                        cutoff_rank = round_cutoffs[category]
                        round_used = round_name
                        break
            
            if cutoff_rank and student_rank <= cutoff_rank:
                eligible_colleges.append({
                    'college_code': college['collegeCode'],
                    'college_name': college['collegeName'],
                    'location': college['location'],
                    'city': college['city'],
                    'cutoff_rank': cutoff_rank,
                    'round_used': round_used,
                    'safety_margin': cutoff_rank - student_rank,
                    'safety_level': self.calculate_safety_level(student_rank, cutoff_rank)
                })
        
        # Sort by cutoff rank (best colleges first)
        eligible_colleges.sort(key=lambda x: x['cutoff_rank'])
        return eligible_colleges
    
    def calculate_safety_level(self, student_rank, cutoff_rank):
        """Calculate safety level based on rank difference"""
        difference = cutoff_rank - student_rank
        if difference > 1500: return 'Very Safe'
        elif difference > 800: return 'Safe'
        elif difference > 300: return 'Moderate'
        elif difference > 0: return 'Competitive'
        else: return 'Unlikely'
    
    def get_round_wise_analysis(self, college_code, category):
        """Get cutoff trends across rounds for a college"""
        college = next((c for c in self.colleges_data if c['collegeCode'] == college_code), None)
        if not college:
            return None
        
        rounds_data = college.get('rounds', {})
        analysis = {}
        
        for round_name, cutoffs in rounds_data.items():
            if category in cutoffs and cutoffs[category] is not None:
                analysis[round_name] = cutoffs[category]
        
        return analysis
    
    def get_statistics_advanced(self):
        """Get comprehensive statistics"""
        total_colleges = len(self.colleges_data)
        cities = set()
        rounds_available = set()
        categories_with_data = set()
        
        for college in self.colleges_data:
            cities.add(college.get('city', 'Unknown'))
            
            # Check primary cutoffs
            for category, cutoff in college.get('cutoffs', {}).items():
                if cutoff is not None:
                    categories_with_data.add(category)
            
            # Check round data
            for round_name in college.get('rounds', {}).keys():
                rounds_available.add(round_name)
        
        return {
            'total_colleges': total_colleges,
            'unique_cities': len(cities),
            'available_rounds': list(rounds_available),
            'categories_with_data': list(categories_with_data),
            'cities': list(cities)
        }

# Test the enhanced handler
if __name__ == "__main__":
    handler = EnhancedPGCETDataHandler()
    stats = handler.get_statistics_advanced()
    print("📊 Enhanced Statistics:", stats)
    
    # Test search
    results = handler.search_by_rank_advanced(3000, 'GM')
    print(f"🔍 Found {len(results)} colleges for rank 3000, GM category")
