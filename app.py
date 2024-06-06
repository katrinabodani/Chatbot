from flask import Flask, render_template, request
from nltk.tokenize import word_tokenize
from nltk import ne_chunk, pos_tag
from nltk import Tree
from nltk.corpus import stopwords
import pandas as pd
import sqlite3
import re


def preprocess_query(text):
    # Pattern to find 'k' notations (e.g., 20k) and convert them to thousands (e.g., 20000)
    def convert_k_to_thousand(match):
        # Extract the numeric part and convert 'k' to thousand
        return str(int(float(match.group(1)) * 1000))
    # Replace 'k' notations with their thousand equivalents
    text = re.sub(r'(\d+\.?\d*)k', convert_k_to_thousand, text, flags=re.IGNORECASE)
    # Add spaces around hyphens if they are directly between numbers to help with NLTK POS Tagging
    text = re.sub(r'(\d+)-(\d+)', r'\1 - \2', text)

    return text


def extract_entities(text):
    # Tokenize and POS tag the text
    words = word_tokenize(text)
    pos_tags = pos_tag(words)

    # Chunk and recognize named entities
    tree = ne_chunk(pos_tags)
    
    # Extract and return entities from the tree
    entities = {'ORGANIZATION': [], 'MONEY': []}
    for subtree in tree:            
        if isinstance(subtree, Tree):  # Check if it's a named entity
            entity = " ".join(word for word, tag in subtree.leaves())  # Join words of multi-word entities
            entity_type = subtree.label()

            # Upon reviewing ive noticed that some phones are being tagged as person so manually tagging them as org
            if entity_type == 'PERSON':
                entity_type = 'ORGANIZATION'

            # Append the entity to the correct list
            if entity_type in entities:
                entities[entity_type].append(entity)
        else:
            # Check for numeric entities (potential prices)
            word, tag = subtree
            if tag == 'CD' and not re.search(r'\d+[A-Za-z]+', word):  # CD tag is used for numeric tokens in POS tagging
                entities['MONEY'].append(word)
    
    # Getting specs keywords            
    specs_keywords = ['GB', 'gb', 'MP', 'mp', 'PTA', 'pta', 'camera', 'RAM' ,'ram','ROM', 'rom', 'mah', 'mAh', 'MAH']
    specs = ' '.join(word for word in text.split() if word != 'compare' and any(keyword in word for keyword in specs_keywords))
    # Extract rating
    rating_match = re.search(r'(\d+(\.\d+)?)\s*-?\s*star', text, re.IGNORECASE)
    if rating_match:
        rating = float(rating_match.group(1))
    else:
        rating_match = re.search(r'rating (?:above|over|of|at least|of above) (\d+(\.\d+)?)', text, re.IGNORECASE)
        rating = float(rating_match.group(1)) if rating_match else ""

    entities['SPECS'] = specs
    entities['RATING'] = rating
    
    return entities

#SQL queries to help get response for chatbot

def get_best_phones_under(price_limit):
    conn = sqlite3.connect('products_reviews.db')
    query = '''
        SELECT Name, Rating, Price 
        FROM Products 
        WHERE Price <= ? 
        ORDER BY Rating DESC, Price
        LIMIT 5
    '''
    df = pd.read_sql_query(query, conn, params=(price_limit,))
    conn.close()

    if not df.empty:
        response = f"Based on user ratings, the best phones under {price_limit} are:\n"
        for _, row in df.iterrows():
            response += f"- {row['Name']} (Rating: {row['Rating']}, Price: {row['Price']})\n"
    else:
        response = f"No phones found under {price_limit}."

    return response

def get_best_phones_in_between(lower_limit,upper_limit):
    conn = sqlite3.connect('products_reviews.db')
    query = '''
        SELECT Name, Rating, Price 
        FROM Products 
        WHERE Price >= ? AND Price <= ? 
        ORDER BY Rating DESC, Price
        LIMIT 5
    '''
    df = pd.read_sql_query(query, conn, params=(lower_limit,upper_limit,))
    conn.close()

    if not df.empty:
        response = f"Based on user ratings, the best phones in price {lower_limit}-{upper_limit} are:\n"
        for _, row in df.iterrows():
            response += f"- {row['Name']} (Rating: {row['Rating']}, Price: {row['Price']})\n"
    else:
        response = f"No phones found in price {lower_limit}-{upper_limit}."

    return response


def get_product_details(product_name):
    conn = sqlite3.connect('products_reviews.db')
    query = '''
        SELECT Name, Price, Rating, Brand, ShippingStatus 
        FROM Products 
        WHERE Name LIKE ?
    '''
    df = pd.read_sql_query(query, conn, params=('%' + product_name + '%',))
    conn.close()

    if not df.empty:
        response = f"Details for {product_name}:\n"
        for _, row in df.iterrows():
            response += f"- Name: {row['Name']}\n  Price: {row['Price']}\n  Rating: {row['Rating']}\n  Brand: {row['Brand']}\n  Shipping Status: {row['ShippingStatus']}\n"
    else:
        response = f"No details found for {product_name}."

    return response



def list_products_by_brand(brand_name):
    conn = sqlite3.connect('products_reviews.db')
    query = '''
        SELECT Name, Price, Rating 
        FROM Products 
        WHERE Brand LIKE ?
    '''
    df = pd.read_sql_query(query, conn, params=('%' + brand_name + '%',))
    conn.close()

    if not df.empty:
        response = f"Products by {brand_name}:\n"
        for _, row in df.iterrows():
            response += f"- {row['Name']} (Price: {row['Price']}, Rating: {row['Rating']})\n"
    else:
        response = f"No products found for brand {brand_name}."

    return response


def compare_products_by_brand(brand_name1, brand_name2):
    conn = sqlite3.connect('products_reviews.db')
    
    query1 = '''    
        SELECT Name, Price, Rating FROM Products WHERE Brand LIKE ?
    '''
    df1 = pd.read_sql_query(query1, conn, params=('%' + brand_name1 + '%',))

    query2 = '''
        SELECT Name, Price, Rating FROM Products WHERE Brand LIKE ?
    '''
    df2 = pd.read_sql_query(query2, conn, params=('%' + brand_name2 + '%',))
    
    conn.close()

    response = ""
    if not df1.empty or not df2.empty:
        response += f"Comparing {brand_name1} and {brand_name2} Products:\n"

        if not df1.empty:
            response += f"\n{brand_name1} Products:\n"
            for _, row in df1.iterrows():
                response += f"- {row['Name']} (Price: {row['Price']}, Rating: {row['Rating']})\n"

        if not df2.empty:
            response += f"\n{brand_name2} Products:\n"
            for _, row in df2.iterrows():
                response += f"- {row['Name']} (Price: {row['Price']}, Rating: {row['Rating']})\n"
    else:
        response = f"No products found for brands: {brand_name1} and/or {brand_name2}."

    return response

def get_phones_with_specs(specs):
    conn = sqlite3.connect('products_reviews.db')
    query = '''
        SELECT Name, Rating, Price 
        FROM Products 
        WHERE Name LIKE ?
        ORDER BY Rating DESC, Price
    '''
    df = pd.read_sql_query(query, conn, params=('%' + specs + '%',))
    conn.close()

    if not df.empty:
        response = f"Phones matching '{specs}':\n"
        for _, row in df.iterrows():
            response += f"- {row['Name']} (Rating: {row['Rating']}, Price: Rs.{row['Price']})\n"
    else:
        response = f"No phones found matching '{specs}'."

    return response

def get_phones_with_multiple_conditions(brand=None, min_rating=None, specs=None, min_price=None, max_price=None):
    conn = sqlite3.connect('products_reviews.db')

    # Building our query
    query = "SELECT Name, Rating, Price FROM Products WHERE 1=1" # base query
    params = []
    criteria = []

    if brand:
        query += " AND Brand = ?"
        params.append(brand)
        criteria.append(f"Brand: {brand}")
    if min_rating:
        query += " AND Rating >= ?"
        params.append(min_rating)
        criteria.append(f"Minimum Rating: {min_rating}")
    if specs:
        query += " AND Name LIKE ?"  # Assuming RAM is part of the product name
        params.append(f"%{specs}%")
        criteria.append(f"Specs: {specs}")
    if min_price:
        query += " AND Price >= ?"
        params.append(min_price)
        criteria.append(f"Minimum Price: Rs.{min_price}")
    if max_price:
        query += " AND Price <= ?"
        params.append(max_price)
        criteria.append(f"Maximum Price: Rs.{max_price}")

    query += " ORDER BY Rating DESC, Price LIMIT 5"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    response = "Top phones based on your criteria:\n"
    if criteria:
        response += "Criteria: " + "; ".join(criteria) + "\n"
    if not df.empty:
        for _, row in df.iterrows():
            response += f"- {row['Name']} (Rating: {row['Rating']}, Price: Rs.{row['Price']})\n"
    else:
        response += "No phones found based on your criteria."

    return response




def process_query(query):
    """
    Process the user query to extract key information and intent.
    """
    
    query = preprocess_query(query)
    # Named Entity Recognition
    entities = extract_entities(query)
    # Intent detection based on keywords
    
    # Handle complex queries with multiple conditions
    if any([entities['ORGANIZATION'], entities['RATING'], entities['SPECS']]):
        unique_non_empty_keys = set()
        # Check and add keys with non-empty values to the set
        if entities.get('ORGANIZATION'):
            unique_non_empty_keys.add('ORGANIZATION')
        if entities.get('RATING'):
            unique_non_empty_keys.add('RATING')
        if entities.get('SPECS'):
            unique_non_empty_keys.add('SPECS')
        if entities.get('MONEY'):
            unique_non_empty_keys.add('MONEY')   
        if len(unique_non_empty_keys) >= 2: # checking for multiple conditons, we dont want the query to run on only single conditions
            print(entities)
            print(unique_non_empty_keys)    
            min_price,max_price = None, None
            if entities['MONEY']:
                if len(entities['MONEY'])==2:
                    prices = list(map(float, entities['MONEY']))
                    if prices:
                        # Get the minimum and maximum price values
                        min_price = min(prices)
                        max_price = max(prices)
                else:
                    max_price = max(map(float, entities['MONEY']))

            return get_phones_with_multiple_conditions(
                brand=entities['ORGANIZATION'][0] if (entities.get('ORGANIZATION')) else entities['ORGANIZATION'], 
                min_rating=entities['RATING'], 
                specs=entities['SPECS'], 
                min_price=min_price, 
                max_price=max_price
            )
    # specs-based queries
    if entities['SPECS']:
        return get_phones_with_specs(entities['SPECS'])
    
    # in between price queries
    elif 'phone' in query.lower() and len(entities['MONEY'])==2:
        prices = list(map(float, entities['MONEY']))
        if prices:
            # Get the minimum and maximum price values
            min_price = min(prices)
            max_price = max(prices)
    
            return get_best_phones_in_between(min_price,max_price)
            
    elif 'phone' in query.lower() and entities['MONEY']:
        price_limit = max(map(float, entities['MONEY']))
        return get_best_phones_under(price_limit)

    elif 'compare' in query.lower() and len(entities['ORGANIZATION']) >= 2:
        return compare_products_by_brand(entities['ORGANIZATION'][0], entities['ORGANIZATION'][1])

    elif 'products by' in query.lower() and entities['ORGANIZATION']:
        return list_products_by_brand(entities['ORGANIZATION'][0])

    elif 'about' in query.lower() and entities['ORGANIZATION']:
        return get_product_details(entities['ORGANIZATION'][0])

    else:
        return "I'm not sure how to answer that. Can you please rephrase your question?"

app = Flask(__name__)

# Database connection function
def get_db_connection():
    conn = sqlite3.connect('products_reviews.db')
    conn.row_factory = sqlite3.Row
    return conn

# Home route with dashboard
@app.route('/')
def index():
    conn = get_db_connection()
    total_listings = conn.execute('SELECT COUNT(*) FROM Products').fetchone()[0]
    avg_price = round(conn.execute('SELECT AVG(Price) FROM Products').fetchone()[0])
    avg_rating = round(conn.execute('SELECT AVG(Rating) FROM Products').fetchone()[0],2)
    avg_review_count_query = '''
    SELECT AVG(ReviewCount) FROM (
        SELECT Products.ProductID, COUNT(Reviews.ReviewID) as ReviewCount
        FROM Products
        LEFT JOIN Reviews ON Products.ProductID = Reviews.ProductID
        GROUP BY Products.ProductID
    )
    '''
    avg_review_count = round(conn.execute(avg_review_count_query).fetchone()[0],2)
    conn.close()
    return render_template('index.html', total_listings=total_listings, avg_price=avg_price, avg_rating=avg_rating, avg_review_count=avg_review_count)

@app.route('/search', methods=['GET'])
def search():
    search_query = request.args.get('search', '')  # Get the search query from the URL parameter
    conn = get_db_connection()
    products = conn.execute('SELECT * FROM Products WHERE Name LIKE ?', ('%' + search_query + '%',)).fetchall()
    conn.close()
    return render_template('search_results.html', products=products, search_query=search_query)


# Global variable to keep track of the number of chatbot queries
chatbot_query_count = 0
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    global chatbot_query_count
    if request.method == 'POST':
        chatbot_query_count += 1  # Increment count on each POST request
        query = request.form['query']
        response = process_query(query)
        return render_template('chatbot.html', response=response, chatbot_query_count=chatbot_query_count)
    return render_template('chatbot.html', chatbot_query_count=chatbot_query_count)


# Route for displaying top 5 products
@app.route('/top-products', methods=['GET', 'POST'])
def top_products():
    criterion = request.args.get('criterion', 'Rating')
    conn = get_db_connection()
    query = f'''
        SELECT Name, Price, Rating, Link
        FROM Products
        ORDER BY {criterion} DESC
        LIMIT 5
    '''
    top_products = conn.execute(query).fetchall()
    conn.close()
    return render_template('top_products.html', top_products=top_products, criterion=criterion)


# Main execution point
if __name__ == '__main__':
    app.run(debug=True)