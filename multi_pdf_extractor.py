# Create multi_pdf_extractor.py
import PyPDF2
import re
import json
from collections import defaultdict

class MultiPDFExtractor:
    def __init__(self):
        self.categories = ['1G','1H','2AG','2AH','2BG','2BH','3AG','3AH',
                          '3BG','3BH','GM','GMH','NKN','PH','SCG','SCH',
                          'STG','STH','XD']
        self.combined_data = {}
        
    def extract_from_single_pdf(self, pdf_path, round_name):
        """Extract data from a single PDF"""
        print(f"📄 Processing {pdf_path}...")
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        # Extract college information
        college_pattern = r'(C\d{3})\s+(.+?)\s+([A-Z\s,.\-()&]+)(?=\n)'
        colleges = re.findall(college_pattern, text)
        
        # Extract year from document (if mentioned)
        year_match = re.search(r'PGCET-(\d{4})', text)
        year = int(year_match.group(1)) if year_match else 2022
        
        # Process colleges
        extracted_colleges = {}
        for code, name, location in colleges:
            extracted_colleges[code] = {
                'collegeCode': code,
                'collegeName': name.strip(),
                'location': location.strip(),
                'city': location.split(',')[-1].strip() if ',' in location else location.strip()
            }
        
        # Extract cutoff ranks in blocks
        # This is a simplified extraction - you may need to adjust based on exact PDF format
        rank_blocks = self.extract_cutoff_blocks(text, len(extracted_colleges))
        
        # Combine college info with cutoffs
        college_list = list(extracted_colleges.keys())
        for i, (college_code, college_info) in enumerate(extracted_colleges.items()):
            cutoffs = {}
            if i < len(rank_blocks):
                for j, category in enumerate(self.categories):
                    if j < len(rank_blocks[i]):
                        cutoff_value = rank_blocks[i][j]
                        cutoffs[category] = int(cutoff_value) if cutoff_value.isdigit() else None
                    else:
                        cutoffs[category] = None
            
            college_info['cutoffs'] = cutoffs
            college_info['round'] = round_name
            college_info['year'] = year
            
            # Add to combined data
            if college_code not in self.combined_data:
                self.combined_data[college_code] = college_info
                self.combined_data[college_code]['rounds'] = {}
            
            # Store round-specific cutoffs
            self.combined_data[college_code]['rounds'][round_name] = cutoffs
    
    def extract_cutoff_blocks(self, text, num_colleges):
        """Extract cutoff rank blocks from PDF text"""
        # Find all numeric values and "--" patterns
        rank_pattern = r'(\d{4,5}|--)'
        ranks = re.findall(rank_pattern, text)
        
        # Group ranks into blocks (19 categories per college)
        blocks = []
        categories_per_college = len(self.categories)
        
        # Skip initial ranks (college codes) and extract cutoff blocks
        start_idx = num_colleges
        for i in range(num_colleges):
            block_start = start_idx + (i * categories_per_college)
            block_end = block_start + categories_per_college
            if block_end <= len(ranks):
                blocks.append(ranks[block_start:block_end])
        
        return blocks
    
    def extract_all_pdfs(self, pdf_files):
        """Extract data from all PDF files"""
        round_names = ['First Round', 'Second Round', 'Third Round']
        
        for pdf_file, round_name in zip(pdf_files, round_names):
            try:
                self.extract_from_single_pdf(pdf_file, round_name)
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
        
        # Convert to list format
        final_data = []
        for college_code, college_data in self.combined_data.items():
            # Use first round cutoffs as primary, with fallback to other rounds
            primary_cutoffs = college_data['rounds'].get('First Round', {})
            if not any(primary_cutoffs.values()):
                # If first round is empty, try second round
                primary_cutoffs = college_data['rounds'].get('Second Round', {})
            if not any(primary_cutoffs.values()):
                # If second round is empty, try third round
                primary_cutoffs = college_data['rounds'].get('Third Round', {})
            
            college_data['cutoffs'] = primary_cutoffs
            final_data.append(college_data)
        
        return final_data
    
    def save_combined_data(self, data, filename='combined_pgcet_data.json'):
        """Save combined data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {len(data)} colleges to {filename}")

# Usage
if __name__ == "__main__":
    extractor = MultiPDFExtractor()
    
    # Extract from all three PDFs
    pdf_files = ['first.pdf', 'second.pdf', 'third.pdf']
    combined_colleges = extractor.extract_all_pdfs(pdf_files)
    
    # Save combined data
    extractor.save_combined_data(combined_colleges)
    
    print(f"🎯 Total colleges extracted: {len(combined_colleges)}")
    print(f"📊 Unique locations: {len(set([c['city'] for c in combined_colleges]))}")
