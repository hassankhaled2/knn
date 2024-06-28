#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify,request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json 
app = Flask(__name__)

# Load the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
whole_predection={"type":"t-shirt","master_category":"Apparel","gender":"men"}
types_map = { 
    0: "Accessory Gift Sets",
    1: "Baby Dolls",
    2: "Backpacks",
    3: "Bangles",
    4: "Basketballs",
    5: "Bath Robes",
    6: "Beauty Accessories",
    7: "Belts",
    8: "Blazers",
    9: "Body Lotions",
    10: "Body Washes and Scrubs",
    11: "Booties",
    12: "Boxers",
    13: "Bras",
    14: "Bracelets",
    15: "Briefs",
    16: "Camisoles",
    17: "Capris",
    18: "Caps",
    19: "Casual Shoes",
    20: "Churidars",
    21: "Clothing Sets",
    22: "Clutches",
    23: "Compacts",
    24: "Concealers",
    25: "Cufflinks",
    26: "Cushion Covers",
    27: "Deodorants",
    28: "Dresses",
    29: "Duffel Bags",
    30: "Dupattas",
    31: "Earrings",
    32: "Eye Creams",
    33: "Eyeshadows",
    34: "Face Moisturisers",
    35: "Face Scrubs and Exfoliators",
    36: "Face Serums and Gels",
    37: "Face Washes and Cleansers",
    38: "Flats",
    39: "Flip Flops",
    40: "Footballs",
    41: "Formal Shoes",
    42: "Foundations and Primers",
    43: "Fragrance Gift Sets",
    44: "Free Gifts",
    45: "Gloves",
    46: "Hair Accessories",
    47: "Hair Colours",
    48: "Handbags",
    49: "Hats",
    50: "Headbands",
    51: "Heels",
    52: "Highlighters and Blushes",
    53: "Innerwear Vests",
    54: "iPads",
    55: "Jackets",
    56: "Jeans",
    57: "Jeggings",
    58: "Jewellery Sets",
    59: "Jumpsuits",
    60: "Kajals and Eyeliners",
    61: "Keychains",
    62: "Kurta Sets",
    63: "Kurtas",
    64: "Kurtis",
    65: "Laptop Bags",
    66: "Leggings",
    67: "Lehenga Cholis",
    68: "Lip Cares",
    69: "Lip Glosses",
    70: "Lip Liners",
    71: "Lip Plumpers",
    72: "Lipsticks",
    73: "Lounge Pants",
    74: "Lounge Shorts",
    75: "Lounge T-shirts",
    76: "Makeup Removers",
    77: "Mascaras",
    78: "Masks and Peels",
    79: "Men's Grooming Kits",
    80: "Messenger Bags",
    81: "Mobile Pouches",
    82: "Mufflers",
    83: "Nail Essentials",
    84: "Nail Polishes",
    85: "Necklaces and Chains",
    86: "Nehru Jackets",
    87: "Night Suits",
    88: "Nightdresses",
    89: "Patialas",
    90: "Pendants",
    91: "Perfume and Body Mists",
    92: "Rain Jackets",
    93: "Rain Trousers",
    94: "Rings",
    95: "Robes",
    96: "Rompers",
    97: "Rucksacks",
    98: "Salwars",
    99: "Salwars and Dupattas",
    100: "Sandals",
    101: "Sarees",
    102: "Scarves",
    103: "Shapewear",
    104: "Shirts",
    105: "Shoe Accessories",
    106: "Shoe Laces",
    107: "Shorts",
    108: "Shrugs",
    109: "Skirts",
    110: "Socks",
    111: "Sports Sandals",
    112: "Sports Shoes",
    113: "Stockings",
    114: "Stoles",
    115: "Sunglasses",
    116: "Sunscreens",
    117: "Suspenders",
    118: "Sweaters",
    119: "Sweatshirts",
    120: "Swimwear",
    121: "Tablet Sleeves",
    122: "Ties",
    123: "Ties and Cufflinks",
    124: "Tights",
    125: "Toners",
    126: "Tops",
    127: "Track Pants",
    128: "Tracksuits",
    129: "Travel Accessories",
    130: "Trolley Bags",
    131: "Trousers",
    132: "Trunks",
    133: "T-shirts",
    134: "Tunics",
    135: "Umbrellas",
    136: "Waist Pouches",
    137: "Waistcoats",
    138: "Wallets",
    139: "Watches",
    140: "Water Bottles",
    141: "Wristbands"
}



@app.route('/get_data', methods=['POST'])
def get_data():
        with open('C:/Users/LENOVO/Desktop/knn_model.pickle', 'rb') as f:
            gender_loaded_model = pickle.load(f)
        gender_data = request.data.decode('utf-8')
        # Process the data as needed
        print("######################################################")
        print(gender_data)
        # Sample input for prediction
        sample_input = [json.loads(gender_data).get("data")]

        # Load the TF-IDF vectorizer and fit it on the training data
        with open(r"C:\Users\hassan\Downloads\tfidf_vectorizer.pickle", 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Transform the input using the fitted TF-IDF vectorizer
        X_tfidf = tfidf_vectorizer.transform(sample_input)
        print(X_tfidf)
        gender_prediction = gender_loaded_model.predict(X_tfidf)
        print(gender_prediction)
        if(gender_prediction==0):
            whole_predection["gender"]="Boys"
        elif (gender_prediction==1):
            whole_predection["gender"]="Girls"
        elif (gender_prediction==2):
            whole_predection["gender"]="Men"
        elif (gender_prediction==3):
             whole_predection["gender"]="Unisex"
        elif (gender_prediction==4):
             whole_predection["gender"]="Women"
                
        print(whole_predection)
                
        print("######################################################")
        with open(r"C:\Users\hassan\Downloads\knn_category_model.pkl", 'rb') as f:
            master_loaded_model = pickle.load(f)
        master_data = request.data.decode('utf-8')
        # Process the data as needed
       
        print(master_data)
        # Sample input for prediction
        sample_input = [json.loads(master_data).get("data")]

        # Load the TF-IDF vectorizer and fit it on the training data
        with open(r"C:\Users\hassan\Downloads\tfidf_vectorizer.pickle", 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Transform the input using the fitted TF-IDF vectorizer
        X_tfidf = tfidf_vectorizer.transform(sample_input)
        print(X_tfidf)
        master_prediction = master_loaded_model.predict(X_tfidf)
        print(master_prediction)
        if(master_prediction==0):
             whole_predection["master_category"]="Accessories"
        elif (master_prediction==1):
            whole_predection["master_category"]="Apparel"
        elif (master_prediction==2):
             whole_predection["master_category"]="Footwear"
        elif (master_prediction==3):
            whole_predection["master_category"]="Free Items"
        elif (master_prediction==4):
             whole_predection["master_category"]="Home"
        elif (master_prediction==5):
            whole_predection["master_category"]="Personal Care"
        elif (master_prediction==6):
            whole_predection["master_category"]="Sporting Goods"
            
        
        with open('C:/Users/LENOVO/Desktop/knn_model.pickle', 'rb') as f:
            type_loaded_model = pickle.load(f)
        type_data = request.data.decode('utf-8')
        # Process the data as needed
       
        print( type_data)
        # Sample input for prediction
        sample_input = [json.loads( type_data).get("data")]

        # Load the TF-IDF vectorizer and fit it on the training data
        with open(r"C:\Users\hassan\Downloads\tfidf_vectorizer.pickle", 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        # Transform the input using the fitted TF-IDF vectorizer
        X_tfidf = tfidf_vectorizer.transform(sample_input)
        print(X_tfidf)
        type_prediction =  type_loaded_model.predict(X_tfidf)
        print(type_prediction)
        whole_predection["type"]=types_map[type_prediction[0]]
        
        
            
        return jsonify(whole_predection)    

if __name__ == '__main__':
    app.run(debug=False)


# In[ ]:


from flask import Flask, jsonify,request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json 
app = Flask(__name__)

# Load the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()


            
#     # Make prediction
#     prediction = loaded_model.predict(X_tfidf)
#     print(prediction)
#     return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=False)


# In[ ]:

#0 Accessories 
#1 Apparel 
#2 Footwear 
#3 Free Items 
#4 Home 
#5 Personal Care 
#6 Sporting Goods 


# In[ ]:


from flask import Flask, jsonify, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json 

app = Flask(__name__)

# Load the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

whole_prediction = {"type": "t-shirt", "master_category": "Apparel", "gender": "men"}

types_map = {
    0: "Accessory Gift Sets", 1: "Baby Dolls", 2: "Backpacks", 3: "Bangles", 4: "Basketballs", 5: "Bath Robes",
    6: "Beauty Accessories", 7: "Belts", 8: "Blazers", 9: "Body Lotions", 10: "Body Washes and Scrubs",
    11: "Booties", 12: "Boxers", 13: "Bras", 14: "Bracelets", 15: "Briefs", 16: "Camisoles", 17: "Capris",
    18: "Caps", 19: "Casual Shoes", 20: "Churidars", 21: "Clothing Sets", 22: "Clutches", 23: "Compacts",
    24: "Concealers", 25: "Cufflinks", 26: "Cushion Covers", 27: "Deodorants", 28: "Dresses", 29: "Duffel Bags",
    30: "Dupattas", 31: "Earrings", 32: "Eye Creams", 33: "Eyeshadows", 34: "Face Moisturisers",
    35: "Face Scrubs and Exfoliators", 36: "Face Serums and Gels", 37: "Face Washes and Cleansers",
    38: "Flats", 39: "Flip Flops", 40: "Footballs", 41: "Formal Shoes", 42: "Foundations and Primers",
    43: "Fragrance Gift Sets", 44: "Free Gifts", 45: "Gloves", 46: "Hair Accessories", 47: "Hair Colours",
    48: "Handbags", 49: "Hats", 50: "Headbands", 51: "Heels", 52: "Highlighters and Blushes",
    53: "Innerwear Vests", 54: "iPads", 55: "Jackets", 56: "Jeans", 57: "Jeggings", 58: "Jewellery Sets",
    59: "Jumpsuits", 60: "Kajals and Eyeliners", 61: "Keychains", 62: "Kurta Sets", 63: "Kurtas", 64: "Kurtis",
    65: "Laptop Bags", 66: "Leggings", 67: "Lehenga Cholis", 68: "Lip Cares", 69: "Lip Glosses",
    70: "Lip Liners", 71: "Lip Plumpers", 72: "Lipsticks", 73: "Lounge Pants", 74: "Lounge Shorts",
    75: "Lounge T-shirts", 76: "Makeup Removers", 77: "Mascaras", 78: "Masks and Peels", 79: "Men's Grooming Kits",
    80: "Messenger Bags", 81: "Mobile Pouches", 82: "Mufflers", 83: "Nail Essentials", 84: "Nail Polishes",
    85: "Necklaces and Chains", 86: "Nehru Jackets", 87: "Night Suits", 88: "Nightdresses", 89: "Patialas",
    90: "Pendants", 91: "Perfume and Body Mists", 92: "Rain Jackets", 93: "Rain Trousers", 94: "Rings",
    95: "Robes", 96: "Rompers", 97: "Rucksacks", 98: "Salwars", 99: "Salwars and Dupattas", 100: "Sandals",
    101: "Sarees", 102: "Scarves", 103: "Shapewear", 104: "Shirts", 105: "Shoe Accessories", 106: "Shoe Laces",
    107: "Shorts", 108: "Shrugs", 109: "Skirts", 110: "Socks", 111: "Sports Sandals", 112: "Sports Shoes",
    113: "Stockings", 114: "Stoles", 115: "Sunglasses", 116: "Sunscreens", 117: "Suspenders", 118: "Sweaters",
    119: "Sweatshirts", 120: "Swimwear", 121: "Tablet Sleeves", 122: "Ties", 123: "Ties and Cufflinks",
    124: "Tights", 125: "Toners", 126: "Tops", 127: "Track Pants", 128: "Tracksuits", 129: "Travel Accessories",
    130: "Trolley Bags", 131: "Trousers", 132: "Trunks", 133: "T-shirts", 134: "Tunics", 135: "Umbrellas",
    136: "Waist Pouches", 137: "Waistcoats", 138: "Wallets", 139: "Watches", 140: "Water Bottles", 141: "Wristbands"
}

@app.route('/get_data_gender', methods=['POST'])
def get_data_gender():
    with open("C:/Users/LENOVO/Desktop/knn_model.pickle", 'rb') as f:
        loaded_model = pickle.load(f)
    
    data = request.data.decode('utf-8')
    # Process the data as needed
    print("######################################################")
    print(data)
    # Sample input for prediction
    sample_input = [json.loads(data).get("data")]

    # Load the TF-IDF vectorizer and fit it on the training data
    with open(r"C:\Users\hassan\Downloads\tfidf_vectorizer.pickle", 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # Transform the input using the fitted TF-IDF vectorizer
    X_tfidf = tfidf_vectorizer.transform(sample_input)
    print(X_tfidf)
    prediction = loaded_model.predict(X_tfidf)
    print(prediction)
    
    if prediction == 0:
        return jsonify({"prediction": "Boys"})
    elif prediction == 1:
        return jsonify({"prediction": "Girls"})
    elif prediction == 2:
        return jsonify({"prediction": "Men"})
    elif prediction == 3:
        return jsonify({"prediction": "Unisex"})
    elif prediction == 4:
        return jsonify({"prediction": "Women"})

@app.route('/get_data_cat', methods=['POST'])
def get_data_cat():
    with open(r"C:\Users\hassan\Downloads\tfidf_vectorizer.pickle", 'rb') as f:
        type_loaded_model = pickle.load(f)
        
    type_data = request.data.decode('utf-8')
    # Process the data as needed

    print(type_data)
    # Sample input for prediction
    sample_input = [json.loads(type_data).get("data")]

    # Load the TF-IDF vectorizer and fit it on the training data
    with open(r"C:\Users\hassan\Downloads\tfidf_vectorizer.pickle", 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    # Transform the input using the fitted TF-IDF vectorizer
    X_tfidf = tfidf_vectorizer.transform(sample_input)
    print(X_tfidf)
    type_prediction = type_loaded_model.predict(X_tfidf)
    
    if type_prediction in types_map:
        whole_prediction["type"] = types_map[type_prediction]
    else:
        whole_prediction["type"] = "Unknown"

if __name__ == '__main__':
    app.run(debug=False)


# In[ ]:




