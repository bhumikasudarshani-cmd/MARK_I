import pandas as pd

def build_final_dataset():
    # 1. Expanded Manual Dataset (Balanced Fake vs. Verified Claims)
    fake_claims = [
        "Drinking turmeric water every morning cures all types of cancer permanently",
        "Neem leaves can completely cure diabetes in just 7 days without medicine",
        "Giloy juice destroys coronavirus instantly when consumed daily",
        "Mixing honey and cinnamon cures arthritis and joint pain forever",
        "Eating raw garlic on empty stomach cures high blood pressure permanently",
        "Homeopathy medicine removes kidney stones without any surgery",
        "Applying coconut oil cures all types of skin infections and fungus",
        "Drinking cow urine daily boosts immunity and prevents all diseases",
        "Alkaline water neutralizes body acidity and reverses chronic illnesses",
        "Magnetic rings worn on the wrist can eliminate joint pain and arthritis",
        "Breathing exercises alone can replace insulin injections for severe diabetes",
        "Charcoal teeth whitening powders permanently remove enamel stains safely overnight",
        "Essential oils applied topically can completely eradicate thyroid disorders",
        "Raw juice fasting for three days detoxifies the liver and cures hepatitis",
        "Colloidal silver safely treats all viral infections without side effects"
    ]

    real_claims = [
        "Paracetamol reduces fever and mild to moderate pain when taken as prescribed",
        "Regular exercise for 30 minutes daily helps maintain healthy blood pressure",
        "Vitamin C supplements support immune function but do not cure colds",
        "Metformin is a commonly prescribed medication for type 2 diabetes management",
        "Hand washing with soap for 20 seconds reduces risk of infection significantly",
        "Antibiotics should be taken for the full prescribed course to prevent resistance",
        "WHO recommends vaccines as one of the most effective disease prevention tools",
        "A balanced diet rich in fruits and vegetables supports overall health",
        "Oral rehydration salts (ORS) effectively manage dehydration caused by acute diarrhea",
        "Insulin therapy is essential for managing type 1 diabetes mellitus under medical supervision",
        "Folic acid supplementation during pregnancy helps prevent neural tube defects in newborns",
        "Statins are clinically proven medications used to lower blood cholesterol levels and reduce cardiovascular risks",
        "Aspirin acts as an antiplatelet agent that can help prevent blood clots when prescribed for heart conditions",
        "Amoxicillin is a broad-spectrum penicillin antibiotic used to treat bacterial infections such as pneumonia and bronchitis",
        "Regular monitoring of blood glucose levels helps patients with diabetes maintain glycemic control",
        "Hypertension is managed through a combination of prescribed antihypertensive drugs and lifestyle modifications",
        "The Central Drugs Standard Control Organization (CDSCO) regulates pharmaceutical imports, approvals, and clinical trials in India",
        "The Indian Council of Medical Research (ICMR) formulates national health guidelines and conducts biomedical research",
        "Vaccination schedules established by health authorities protect populations against preventable infectious diseases",
        "Proper storage of medications away from heat and moisture ensures their chemical stability and efficacy",
        "Generic drugs contain the same active pharmaceutical ingredients and meet the same quality standards as brand-name drugs",
        "Antihistamines are commonly used to relieve symptoms of allergic rhinitis and seasonal allergies",
        "Topical antifungal creams effectively treat localized skin infections like ringworm and athlete's foot",
        "Iron supplements are prescribed to treat iron deficiency anemia and improve hemoglobin levels",
        "Inhaled corticosteroids are standard maintenance therapies for controlling inflammation in asthma patients",
        "Regular dental hygiene including brushing and flossing prevents plaque accumulation and periodontal disease",
        "Sufficient daily water intake is necessary for maintaining optimal cellular function and metabolic processes",
        "Screening tests like mammograms and pap smears aid in the early detection of specific cancers",
        "Completed immunizations are vital for achieving community herd immunity against viral and bacterial pathogens",
        "Consulting a licensed healthcare professional ensures safe and accurate diagnosis before initiating any drug treatment"
    ]

    # Combining manual text and labels into a dataframe
    manual_texts = fake_claims + real_claims
    manual_labels = [0] * len(fake_claims) + [1] * len(real_claims)
    
    manual_data = pd.DataFrame({
        'text': manual_texts,
        'label': manual_labels
    })

    # 2. Load Government Drug Alert CSV (Labeled as Fake/Substandard = 0)
    alert_df = pd.read_csv('Drug_Alert_March2023.csv')
    alert_df['text'] = alert_df['Name of Drugs/medical device/cosmetic'].astype(str) + " - Failure noted: " + alert_df['Reason for failure'].astype(str)
    alert_df = alert_df[['text']]
    alert_df['label'] = 0

    # 3. Load Spurious Drugs CSV (Labeled as Fake/Counterfeit = 0)
    spurious_df = pd.read_csv('Spurious_Drugs_Puducherry_Nov2025.csv')
    spurious_df['text'] = spurious_df['Name of the Tablet'].astype(str) + " - Notice: " + spurious_df['Reason for declared Spurious/ Counterfeit'].astype(str)
    spurious_df = spurious_df[['text']]
    spurious_df['label'] = 0
    
    # 4. Concatenate all dataframes into one master dataframe
    frames = [manual_data, alert_df, spurious_df] 
    final_df = pd.concat(frames, ignore_index=True)

    # 5. Clean the dataset
    final_df = final_df.dropna(subset=['text'])           # Drop empty rows
    final_df = final_df.drop_duplicates(subset=['text'])  # Drop duplicate records

    # 6. Shuffle dataset rows to ensure proper class distribution across splits
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 7. Export to CSV for your training pipeline
    final_df.to_csv('claims.csv', index=False)
    
    print("Dataset successfully expanded, compiled, and saved to claims.csv!")
    print(f"Total samples: {len(final_df)}")
    print(final_df['label'].value_counts())

if __name__ == "__main__":
    build_final_dataset()