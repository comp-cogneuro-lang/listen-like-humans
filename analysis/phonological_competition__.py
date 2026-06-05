def DAS_Neighborhood_Checker(word1, word2, pronunciation_Dict):
    """Check if two words are DAS neighbors (differ by one phoneme operation)."""
    pronunciation1 = pronunciation_Dict[word1]["pronunciation"]
    pronunciation2 = pronunciation_Dict[word2]["pronunciation"]

    # Same word
    if word1 == word2:
        return False

    # Exceed range
    elif abs(len(pronunciation1) - len(pronunciation2)) > 1:
        return False

    # Deletion
    elif len(pronunciation1) == len(pronunciation2) + 1:
        for index in range(len(pronunciation1)):
            deletion = pronunciation1[:index] + pronunciation1[index + 1:]
            if deletion == pronunciation2:
                return True

    # Addition
    elif len(pronunciation1) == len(pronunciation2) - 1:
        for index in range(len(pronunciation2)):
            deletion = pronunciation2[:index] + pronunciation2[index + 1:]
            if deletion == pronunciation1:
                return True

    # Substitution
    elif len(pronunciation1) == len(pronunciation2):
        for index in range(len(pronunciation1)):
            pronunciation1_Substitution = pronunciation1[:index] + pronunciation1[index + 1:]
            pronunciation2_Substitution = pronunciation2[:index] + pronunciation2[index + 1:]
            if pronunciation1_Substitution == pronunciation2_Substitution:
                return True

    return False


def Category_Dict_Generate(using_Word_List, pronunciation_Dict, w2v_model=None):
    """Generate category dictionary for phonological competition analysis."""
    print("Category dict generating...")
    num_semantic_nearbors = 5
    category_Dict = {}
    
    import hashlib, os
    hash_code = hashlib.md5(str(using_Word_List).encode()).hexdigest()
    if os.path.exists(f"cache/category_dict_{hash_code}.pkl"):
        import pickle
        with open(f"cache/category_dict_{hash_code}.pkl", "rb") as f:
            category_Dict = pickle.load(f)
        print("Category dict loaded from cache.")
        return category_Dict
    
    for target_Word in using_Word_List:
        target_Pronunciation = pronunciation_Dict[target_Word]["pronunciation"]

        category_Dict[target_Word, "Target"] = []
        category_Dict[target_Word, "Cohort"] = []
        category_Dict[target_Word, "Rhyme"] = []
        category_Dict[target_Word, "Embedding"] = []
        category_Dict[target_Word, "DAS_Neighborhood"] = []
        if w2v_model:
            semantic_neighborhood = [word for word, sim in w2v_model.most_similar_cosmul(target_Word + "-en", topn=num_semantic_nearbors)]
            semantic_neighborhood = semantic_neighborhood[:num_semantic_nearbors]
            category_Dict[target_Word, "Semantic-top3"] = [ word.split("-")[0] for word in semantic_neighborhood[:3]]
            category_Dict[target_Word, "Semantic-top5"] = [ word.split("-")[0] for word in semantic_neighborhood[:5]]
        for compare_Word in using_Word_List:
            compare_Pronunciation = pronunciation_Dict[compare_Word]["pronunciation"]

            unrelated = True
            if target_Word == compare_Word:
                category_Dict[target_Word, "Target"].append(compare_Word)
                unrelated = False
            if target_Pronunciation[0:2] == compare_Pronunciation[0:2] and target_Word != compare_Word:
                category_Dict[target_Word, "Cohort"].append(compare_Word)
                unrelated = False
            if target_Pronunciation[1:] == compare_Pronunciation[1:] and target_Pronunciation[0] != compare_Pronunciation[0] and target_Word != compare_Word:
                category_Dict[target_Word, "Rhyme"].append(compare_Word)
                unrelated = False
            if compare_Pronunciation in target_Pronunciation and target_Word != compare_Word:
                category_Dict[target_Word, "Embedding"].append(compare_Word)
                unrelated = False

            # if unrelated:
            #     category_Dict[target_Word, "Unrelated"].append(compare_Word)
            # For test
            if DAS_Neighborhood_Checker(target_Word, compare_Word, pronunciation_Dict):
                category_Dict[target_Word, "DAS_Neighborhood"].append(compare_Word)
    import pickle
    with open(f"cache/category_dict_{hash_code}.pkl", "wb") as f:
        pickle.dump(category_Dict, f)
    print("Category dict saved to cache.")
    return category_Dict


import numpy as np
def RT_Dict_Generate_item(
    epoch,
    word,
    cs_array,
    talker,
    word_Index_Dict,
    enes_map,
    esen_map,
    max_Cycle,
    absolute_Criterion=0.7,
    relative_Criterion=0.05,
    time_Dependency_Criterion=(10, 0.05)
):
    """
    Generate reaction time dictionary for a given word and talker.
    
    Args:
        epoch: Epoch number
        word: Target word
        cs_array: Competition score array
        talker: Talker identifier
        word_Index_Dict: Dictionary mapping words to their indices
        enes_map: English-Spanish word mapping
        esen_map: Spanish-English word mapping
        max_Cycle: Maximum number of cycles to check
        absolute_Criterion: Absolute threshold criterion (default: 0.7)
        relative_Criterion: Relative threshold criterion (default: 0.05)
        time_Dependency_Criterion: Time dependency criterion tuple (default: (10, 0.05))
    
    Returns:
        rt_Dict: Dictionary containing reaction time results
    """
    rt_Dict = {}
    target_Index = word_Index_Dict[word]
    synomym_Index = -1
    
    if enes_map:
        if word in enes_map.keys():
            synomym_Index = word_Index_Dict[enes_map[word]]
        elif word in esen_map.keys():
            synomym_Index = word_Index_Dict[esen_map[word]]
        else:
            raise Exception("No synomym found for word: {}".format(word))
    
        assert abs(synomym_Index - target_Index) == 1, "Synomym index should not be adjacent to target index."
    target_Array = cs_array[target_Index]
    if synomym_Index != -1:
        other_Max_Array = np.max(np.delete(cs_array, [target_Index, synomym_Index], 0), axis=0)
    else:
        other_Max_Array = np.max(np.delete(cs_array, target_Index, 0), axis=0)
            
    # Absolute threshold RT
    if not (other_Max_Array > absolute_Criterion).any():
        absolute_Check_Array = target_Array > absolute_Criterion
        for cycle in range(max_Cycle):
            if absolute_Check_Array[cycle]:
                rt_Dict["Absolute", epoch, word, talker] = cycle
                break
    if not ("Absolute", epoch, word, talker) in rt_Dict.keys():
        rt_Dict["Absolute", epoch, word, talker] = np.nan
    
    # Relative threshold RT
    relative_Check_Array = target_Array > (other_Max_Array + relative_Criterion)
    for cycle in range(max_Cycle):
        if relative_Check_Array[cycle]:
            rt_Dict["Relative", epoch, word, talker] = cycle
            break
    if not ("Relative", epoch, word, talker) in rt_Dict.keys():
        rt_Dict["Relative", epoch, word, talker] = np.nan
    
    # Time dependent RT
    time_Dependency_Check_Array_with_Criterion = target_Array > other_Max_Array + time_Dependency_Criterion[1]
    time_Dependency_Check_Array_Sustainment = target_Array > other_Max_Array
    for cycle in range(max_Cycle - time_Dependency_Criterion[0]):
        if all(np.hstack([
            time_Dependency_Check_Array_with_Criterion[cycle:cycle + time_Dependency_Criterion[0]],
            time_Dependency_Check_Array_Sustainment[cycle + time_Dependency_Criterion[0]:]
        ])):
            rt_Dict["Time_Dependent", epoch, word, talker] = cycle
            break
    if not ("Time_Dependent", epoch, word, talker) in rt_Dict.keys():
        rt_Dict["Time_Dependent", epoch, word, talker] = np.nan
    
    return rt_Dict


import pandas as pd
import numpy as np

def dict_to_dataframe(categorized_data_dict):
    """
    Convert dictionary with structure categorized_data_dict[epoch, word, speaker, category] = np.array([...])
    to pandas DataFrame with columns: Epoch, Word, Speaker, Category, 0, 1, 2, 3, ...
    """
    
    # Find maximum array length
    max_len = max(len(v) for v in categorized_data_dict.values())
    
    # Prepare data for DataFrame
    rows = []
    for (epoch, word, speaker, category), array in categorized_data_dict.items():
        # Pad array with np.nan to max_len
        padded_array = np.pad(array, (0, max_len - len(array)), constant_values=np.nan)
        
        # Create row with metadata and array values
        row = {
            'Epoch': epoch,
            'Word': word,
            'Speaker': speaker,
            'Category': category
        }
        
        # Add array values as columns 0, 1, 2, ...
        for idx, val in enumerate(padded_array):
            row[idx] = val
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    return df