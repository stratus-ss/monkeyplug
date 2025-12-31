import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Set

def load_swear_words(swear_file_path: str) -> Set[str]:
    """
    Load swear words from JSON file.
    Handles various JSON formats (array, object with words key, etc.)
    """
    with open(swear_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        swear_words = set(word.lower().strip() for word in data)
    elif isinstance(data, dict):
        # Common keys: 'words', 'profanity', 'swears', 'list'
        for key in ['words', 'profanity', 'swears', 'list', 'items']:
            if key in data:
                swear_words = set(word.lower().strip() for word in data[key])
                break
        else:
            # If no recognized key, use all values
            swear_words = set(str(v).lower().strip() for v in data.values())
    else:
        raise ValueError(f"Unsupported JSON format in {swear_file_path}")
    
    return swear_words

def find_profanity_in_transcript(transcript_data: List[Dict], swear_words: Set[str]) -> List[Dict]:
    """
    Find all profanity instances in a transcript.
    Returns list of matching words with their metadata.
    """
    profanity_matches = []
    
    for word_data in transcript_data:
        word = word_data.get('word', '').lower().strip()
        # Remove common punctuation for matching
        word_clean = word.strip('.,!?;:\'"')
        
        if word_clean in swear_words:
            match = {
                'word': word_data.get('word'),
                'word_clean': word_clean,
                'start': word_data.get('start'),
                'end': word_data.get('end'),
                'confidence': word_data.get('conf'),
            }
            profanity_matches.append(match)
            
            # Mark in original data
            word_data['scrub'] = True
    
    return profanity_matches

def process_transcript_file(transcript_path: str, swear_words: Set[str]) -> Dict:
    """
    Process a single transcript file and return profanity report.
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    profanity_matches = find_profanity_in_transcript(transcript_data, swear_words)
    
    return {
        'file': transcript_path,
        'total_words': len(transcript_data),
        'profanity_count': len(profanity_matches),
        'matches': profanity_matches
    }

def process_all_transcripts(transcript_dir: str, swear_file: str, output_file: str = None):
    """
    Process all transcript JSON files in a directory.
    """
    # Load swear words
    swear_words = load_swear_words(swear_file)
    print(f"Loaded {len(swear_words)} swear words from {swear_file}")
    
    # Find all transcript JSON files
    transcript_dir_path = Path(transcript_dir)
    transcript_files = sorted(transcript_dir_path.glob('*.json'))
    
    if not transcript_files:
        print(f"No JSON files found in {transcript_dir}")
        return
    
    print(f"Found {len(transcript_files)} transcript files")
    
    # Process each transcript
    results = []
    total_profanity = 0
    
    for transcript_file in transcript_files:
        print(f"\nProcessing: {transcript_file.name}")
        result = process_transcript_file(str(transcript_file), swear_words)
        results.append(result)
        total_profanity += result['profanity_count']
        
        print(f"  Found {result['profanity_count']} instances of profanity")
        
        # Print all matches
        if result['matches']:
            print("  All matches:")
            for match in result['matches']:
                print(f"    - '{match['word']}' at {match['start']:.2f}s - {match['end']:.2f}s")
    
    # Generate summary report
    summary = {
        'total_transcripts': len(transcript_files),
        'total_profanity_instances': total_profanity,
        'swear_words_loaded': len(swear_words),
        'results': results
    }
    
    # Save results to output file
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull report saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total transcripts processed: {summary['total_transcripts']}")
    print(f"Total profanity instances: {summary['total_profanity_instances']}")
    print(f"Swear words in dictionary: {len(swear_words)}")
    
    # # Generate FFmpeg filter commands
    # print(f"\n{'='*60}")
    # print(f"FFMPEG MUTE COMMANDS")
    # print(f"{'='*60}")
    
    # for result in results:
    #     if result['profanity_count'] > 0:
    #         audio_file = Path(result['file']).stem + '.m4b'  # Adjust extension as needed
    #         print(f"\n# {audio_file}")
            
    #         # Generate volume filter for muting
    #         filter_parts = []
    #         for match in result['matches']:
    #             start = match['start'] - 0.1  # Add padding before
    #             end = match['end'] + 0.1      # Add padding after
    #             filter_parts.append(f"volume=0:enable='between(t,{start:.2f},{end:.2f})'")
            
    #         filter_chain = ",".join(filter_parts)
            
#            print(f"ffmpeg -i {audio_file} -af \"{filter_chain}\" -c:a copy {Path(audio_file).stem}_clean.m4b")
    
    return summary

def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Find profanity in transcript JSON files and generate a report.'
    )
    
    parser.add_argument(
        '--swear-words-file',
        type=str,
        default='swear_list.json',
        help='Path to the JSON file containing swear words (default: swear_list.json)'
    )
    
    parser.add_argument(
        '--transcript-dir',
        type=str,
        default='output',
        help='Directory containing transcript JSON files (default: output)'
    )
    
    parser.add_argument(
        '--output-report',
        type=str,
        default='profanity_report.json',
        help='Path for the output report JSON file (default: profanity_report.json)'
    )
    
    args = parser.parse_args()
    
    # Process all transcripts and generate report
    summary = process_all_transcripts(
        transcript_dir=args.transcript_dir,
        swear_file=args.swear_words_file,
        output_file=args.output_report
    )
    
    return summary

# Main execution
if __name__ == "__main__":
    main()
