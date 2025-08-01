import json
import random
from collections import defaultdict, Counter
import re

def extract_task_domain(instruction):
    """
    Extract task and domain from instruction string
    Expected format: [Task: NER][Domain: jnlpba/jnlpba]
    """
    task_match = re.search(r'\[Task:\s*([^\]]+)\]', instruction)
    domain_match = re.search(r'\[Domain:\s*([^\]]+)\]', instruction)
    
    task = task_match.group(1).strip() if task_match else "Unknown"
    domain = domain_match.group(1).strip() if domain_match else "Unknown"
    
    return task, domain

def load_and_categorize_data(data):
    """
    Load data and categorize by [Task: **][Domain: **]
    """
    categories = defaultdict(list)
    
    for item in data:
        instruction = item.get('Instruction', '')
        task, domain = extract_task_domain(instruction)
        
        category_key = f"[Task: {task}][Domain: {domain}]"
        categories[category_key].append(item)
    
    return categories

def get_sampling_statistics(categories):
    """
    Get statistics about each category
    """
    stats = {}
    for category, items in categories.items():
        stats[category] = len(items)
    
    return stats

def sample_data(categories, sample_size=None):
    """
    Sample data from each category
    If sample_size is None, use the minimum category size
    """
    if not categories:
        return {}

    # Get category sizes
    category_sizes = {cat: len(items) for cat, items in categories.items()}

    print(category_sizes)
    
    # Determine sample size
    if sample_size is None:
        sample_size = 500 ###########################################################采样量
        print(f"Using minimum category size as sample size: {sample_size}")
    
    # Sample from each category
    sampled_data = {}
    for category, items in categories.items():
        if len(items) >= sample_size:
            sampled_items = random.sample(items, sample_size)
        else:
            sampled_items = items  # Use all items if category has fewer than sample_size
            print(f"Warning: Category {category} has only {len(items)} items, using all")
        
        sampled_data[category] = sampled_items
    
    return sampled_data

def main():
    # Load your actual data here
    # For example: 
    with open('/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/multitask_bio_dataset_all_8_newnew.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Categorize data
    categories = load_and_categorize_data(data)

    # Print statistics
    print("Dataset Statistics:")
    print("-" * 50)
    stats = get_sampling_statistics(categories)
    for category, count in stats.items():
        print(f"{category}: {count} samples")
    
    # Find minimum sample size
    # min_samples = min(stats.values()) if stats else 0
    min_samples = 1000
    print(f"\nMinimum sample size across categories: {min_samples}")
    
    # Sample data
    sampled_data = sample_data(categories)
    
    # Print sampling results
    print("\nSampling Results:")
    print("-" * 50)
    total_sampled = 0
    for category, items in sampled_data.items():
        print(f"{category}: {len(items)} samples")
        total_sampled += len(items)
    
    print(f"\nTotal samples after sampling: {total_sampled}")
    
    # Convert back to list format
    final_sampled_list = []
    for category, items in sampled_data.items():
        final_sampled_list.extend(items)
    
    # Shuffle the final list
    random.shuffle(final_sampled_list)
    
    # Save sampled data
    with open('/home/siat/Data/synthetic/MinerU/paper_parse/multi_task/multitask_bio_dataset_all_sample_8_new_new1.json', 'w', encoding='utf-8') as f:
        json.dump(final_sampled_list, f, ensure_ascii=False, indent=2)
    
    print(f"\nSampled dataset contains {len(final_sampled_list)} total samples")
    
    return final_sampled_list

if __name__ == "__main__":
    sampled_dataset = main()