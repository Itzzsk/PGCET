# Create advanced_ml_predictor.py (CORRECTED VERSION)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import json

class AdvancedPGCETPredictor:
    def __init__(self, data_file='combined_pgcet_data.json'):
        self.data_file = data_file
        self.colleges_data = self.load_data()
        
        # Models
        self.admission_model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.probability_model = GradientBoostingRegressor(n_estimators=150, random_state=42)
        self.cutoff_trend_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Encoders
        self.category_encoder = LabelEncoder()
        self.college_encoder = LabelEncoder()
        self.city_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        self.is_trained = False
        
    def load_data(self):
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: {self.data_file} not found!")
            print("Please run multi_pdf_extractor.py first to create the combined data file.")
            return []
    
    def extract_enhanced_features(self, college):
        """Extract comprehensive features from college data"""
        location = college.get('location', '').upper()
        name = college.get('collegeName', '').upper()
        
        # Get cutoffs from all rounds
        all_cutoffs = []
        for round_data in college.get('rounds', {}).values():
            all_cutoffs.extend([v for v in round_data.values() if v is not None and str(v).isdigit()])
        
        # If no round data, use primary cutoffs
        if not all_cutoffs:
            all_cutoffs = [v for v in college.get('cutoffs', {}).values() if v is not None and str(v).isdigit()]
        
        all_cutoffs = [int(c) for c in all_cutoffs]
        
        features = {
            # Location features
            'is_bangalore': 1 if 'BANGALORE' in location else 0,
            'is_mysore': 1 if 'MYSORE' in location else 0,
            'is_hubli': 1 if 'HUBLI' in location else 0,
            'is_mangalore': 1 if 'MANGALORE' in location else 0,
            'is_tier1_city': 1 if any(city in location for city in ['BANGALORE', 'MYSORE', 'HUBLI', 'MANGALORE']) else 0,
            
            # Institution type
            'is_university': 1 if 'UNIVERSITY' in name else 0,
            'is_institute_tech': 1 if any(word in name for word in ['INSTITUTE OF TECHNOLOGY', 'ENGINEERING', 'TECHNICAL']) else 0,
            'is_college': 1 if 'COLLEGE' in name else 0,
            
            # Prestige indicators
            'is_government': 1 if any(word in name for word in ['UNIVERSITY', 'GOVERNMENT', 'GOVT']) else 0,
            'is_autonomous': 1 if 'AUTONOMOUS' in name else 0,
            
            # Cutoff-based features
            'min_cutoff': min(all_cutoffs) if all_cutoffs else 8000,
            'max_cutoff': max(all_cutoffs) if all_cutoffs else 12000,
            'avg_cutoff': np.mean(all_cutoffs) if all_cutoffs else 10000,
            'cutoff_range': max(all_cutoffs) - min(all_cutoffs) if len(all_cutoffs) > 1 else 0,
            'total_categories_available': len(all_cutoffs),
            
            # Round availability
            'has_multiple_rounds': len(college.get('rounds', {})) > 1,
            'rounds_count': len(college.get('rounds', {}))
        }
        
        return features
    
    def create_comprehensive_training_data(self):
        """Create comprehensive training dataset"""
        training_samples = []
        
        if not self.colleges_data:
            print("❌ No college data available for training!")
            return pd.DataFrame()
        
        for college in self.colleges_data:
            college_features = self.extract_enhanced_features(college)
            
            # Process all rounds or primary cutoffs
            rounds_to_process = college.get('rounds', {})
            if not rounds_to_process:
                # Use primary cutoffs if no rounds data
                rounds_to_process = {'Primary': college.get('cutoffs', {})}
            
            for round_name, round_cutoffs in rounds_to_process.items():
                for category, cutoff in round_cutoffs.items():
                    if cutoff is not None and str(cutoff).isdigit():
                        cutoff = int(cutoff)
                        
                        # Generate positive samples (admitted students)
                        for rank in range(2001, cutoff + 1, 120):
                            sample = {
                                'student_rank': rank,
                                'category': category,
                                'college_code': college['collegeCode'],
                                'round': round_name,
                                'cutoff_rank': cutoff,
                                'gets_admission': 1,
                                'admission_probability': min(0.95, 0.6 + (cutoff - rank) / cutoff * 0.35),
                                **college_features
                            }
                            training_samples.append(sample)
                        
                        # Generate negative samples (not admitted)
                        for rank in range(cutoff + 50, min(cutoff + 1800, 11000), 180):
                            sample = {
                                'student_rank': rank,
                                'category': category,
                                'college_code': college['collegeCode'],
                                'round': round_name,
                                'cutoff_rank': cutoff,
                                'gets_admission': 0,
                                'admission_probability': max(0.05, 0.4 - (rank - cutoff) / cutoff * 0.35),
                                **college_features
                            }
                            training_samples.append(sample)
        
        return pd.DataFrame(training_samples)
    
    def train_models(self):
        """Train all ML models"""
        print("🤖 Creating comprehensive training dataset...")
        training_data = self.create_comprehensive_training_data()
        
        if training_data.empty:
            print("❌ No training data available!")
            return 0, 0
        
        print(f"📊 Generated {len(training_data)} training samples")
        
        # Encode categorical variables
        training_data['category_encoded'] = self.category_encoder.fit_transform(training_data['category'])
        training_data['college_encoded'] = self.college_encoder.fit_transform(training_data['college_code'])
        
        # Prepare features
        feature_columns = [
            'student_rank', 'category_encoded', 'college_encoded', 'cutoff_rank',
            'is_bangalore', 'is_mysore', 'is_hubli', 'is_mangalore', 'is_tier1_city',
            'is_university', 'is_institute_tech', 'is_college', 'is_government',
            'min_cutoff', 'avg_cutoff', 'cutoff_range', 'total_categories_available',
            'has_multiple_rounds', 'rounds_count'
        ]
        
        X = training_data[feature_columns]
        y_admission = training_data['gets_admission']
        y_probability = training_data['admission_probability']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_adm_train, y_adm_test, y_prob_train, y_prob_test = train_test_split(
            X_scaled, y_admission, y_probability, test_size=0.2, random_state=42
        )
        
        # Train admission classifier
        print("🎯 Training admission prediction model...")
        self.admission_model.fit(X_train, y_adm_train)
        adm_accuracy = accuracy_score(y_adm_test, self.admission_model.predict(X_test))
        
        # Train probability regressor
        print("📈 Training probability prediction model...")
        self.probability_model.fit(X_train, y_prob_train)
        prob_mae = mean_absolute_error(y_prob_test, self.probability_model.predict(X_test))
        
        print(f"✅ Admission Model Accuracy: {adm_accuracy:.3f}")
        print(f"✅ Probability Model MAE: {prob_mae:.3f}")
        
        self.is_trained = True
        return adm_accuracy, prob_mae
    
    def predict_with_intelligence(self, student_rank, category, preferences=None):
        """Intelligent prediction with preferences"""
        if not self.is_trained:
            raise ValueError("Models not trained! Call train_models() first.")
        
        predictions = []
        preferences = preferences or {}
        
        for college in self.colleges_data:
            # Check if college has data for this category
            has_category = False
            best_cutoff = None
            best_round = None
            
            # Check all rounds for best cutoff
            rounds_data = college.get('rounds', {})
            if not rounds_data:
                rounds_data = {'Primary': college.get('cutoffs', {})}
            
            for round_name, round_cutoffs in rounds_data.items():
                if category in round_cutoffs and round_cutoffs[category] is not None:
                    cutoff = int(round_cutoffs[category]) if str(round_cutoffs[category]).isdigit() else None
                    if cutoff and student_rank <= cutoff:
                        if best_cutoff is None or cutoff < best_cutoff:
                            best_cutoff = cutoff
                            best_round = round_name
                        has_category = True
            
            if has_category and best_cutoff:
                college_features = self.extract_enhanced_features(college)
                
                try:
                    # Prepare features
                    features = np.array([[
                        student_rank,
                        self.category_encoder.transform([category])[0],
                        self.college_encoder.transform([college['collegeCode']])[0],
                        best_cutoff,
                        college_features['is_bangalore'],
                        college_features['is_mysore'],
                        college_features['is_hubli'],
                        college_features['is_mangalore'],
                        college_features['is_tier1_city'],
                        college_features['is_university'],
                        college_features['is_institute_tech'],
                        college_features['is_college'],
                        college_features['is_government'],
                        college_features['min_cutoff'],
                        college_features['avg_cutoff'],
                        college_features['cutoff_range'],
                        college_features['total_categories_available'],
                        college_features['has_multiple_rounds'],
                        college_features['rounds_count']
                    ]])
                    
                    features_scaled = self.scaler.transform(features)
                    
                    # Get predictions
                    admission_prob = self.admission_model.predict_proba(features_scaled)[0][1]
                    probability_score = max(0, min(1, self.probability_model.predict(features_scaled)[0]))
                    
                    # Combine predictions
                    final_probability = (admission_prob + probability_score) / 2
                    
                    # Apply preferences
                    preference_bonus = 0
                    if preferences.get('preferred_city') and preferences['preferred_city'].upper() in college.get('location', '').upper():
                        preference_bonus += 0.1
                    if preferences.get('prefer_government') and college_features['is_government']:
                        preference_bonus += 0.05
                    if preferences.get('prefer_university') and college_features['is_university']:
                        preference_bonus += 0.05
                    
                    final_probability = min(0.98, final_probability + preference_bonus)
                    
                    predictions.append({
                        'college_code': college['collegeCode'],
                        'college_name': college['collegeName'],
                        'location': college['location'],
                        'city': college.get('city', college['location'].split(',')[-1].strip()),
                        'cutoff_rank': best_cutoff,
                        'best_round': best_round,
                        'admission_probability': final_probability,
                        'safety_level': self.calculate_safety_level(student_rank, best_cutoff),
                        'rank_difference': best_cutoff - student_rank,
                        'college_features': college_features,
                        'preference_match': preference_bonus > 0
                    })
                    
                except (ValueError, KeyError) as e:
                    # Handle unseen categories/colleges
                    print(f"⚠️ Skipping {college['collegeCode']}: {e}")
                    continue
        
        # Sort by probability and preference match
        predictions.sort(key=lambda x: (x['preference_match'], x['admission_probability']), reverse=True)
        return predictions
    
    def calculate_safety_level(self, student_rank, cutoff_rank):
        difference = cutoff_rank - student_rank
        if difference > 1500: return 'Very Safe'
        elif difference > 800: return 'Safe'  
        elif difference > 300: return 'Moderate'
        elif difference > 0: return 'Competitive'
        else: return 'Reach'
    
    def save_models(self, filepath='advanced_pgcet_model.pkl'):
        model_data = {
            'admission_model': self.admission_model,
            'probability_model': self.probability_model,
            'category_encoder': self.category_encoder,
            'college_encoder': self.college_encoder,
            'scaler': self.scaler,
            'colleges_data': self.colleges_data
        }
        joblib.dump(model_data, filepath)
        print(f"🎯 Models saved to {filepath}")
    
    @classmethod
    def load_models(cls, filepath, data_file):
        predictor = cls(data_file)
        try:
            model_data = joblib.load(filepath)
            
            predictor.admission_model = model_data['admission_model']
            predictor.probability_model = model_data['probability_model']
            predictor.category_encoder = model_data['category_encoder']
            predictor.college_encoder = model_data['college_encoder']
            predictor.scaler = model_data['scaler']
            predictor.is_trained = True
            
            print(f"✅ Models loaded from {filepath}")
            return predictor
        except FileNotFoundError:
            print(f"❌ Model file {filepath} not found!")
            return predictor

# Train and test
if __name__ == "__main__":
    print("🚀 Starting Advanced PGCET Predictor Training...")
    
    predictor = AdvancedPGCETPredictor()
    
    if not predictor.colleges_data:
        print("❌ No data available. Please run multi_pdf_extractor.py first!")
        exit(1)
    
    accuracy, mae = predictor.train_models()
    
    if accuracy > 0:
        predictor.save_models()
        
        # Test predictions
        print("\n🎯 Testing predictions...")
        test_predictions = predictor.predict_with_intelligence(
            student_rank=3500, 
            category='GM',
            preferences={'preferred_city': 'BANGALORE', 'prefer_government': True}
        )
        
        print(f"\n🎯 Top 5 predictions for rank 3500, GM category:")
        for i, pred in enumerate(test_predictions[:5], 1):
            print(f"{i}. {pred['college_name'][:50]}")
            print(f"   📍 {pred['city']} | 🎯 Cutoff: {pred['cutoff_rank']}")
            print(f"   📊 Probability: {pred['admission_probability']:.1%} | 🛡️ {pred['safety_level']}")
            print(f"   🔄 Best Round: {pred['best_round']}")
    else:
        print("❌ Training failed! Check your data file.")
