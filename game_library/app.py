from flask import Flask, render_template, request, jsonify
import sqlite3
import json
import random
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('Games.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    with app.open_resource('schema.sql') as f:
        conn.executescript(f.read().decode('utf8'))
    conn.commit()
    conn.close()

def get_data():
    conn = get_db_connection()
    query = "SELECT * FROM played_games"
    data = pd.read_sql_query(query, conn)
    conn.close()
    if data.empty:
        print("No data found in played_games table.")
    return data

def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

def train_model(train_data):
    if train_data.empty:
        raise ValueError("Training data is empty. Cannot train the model.")

    features = train_data[['year', 'criticscore']]  # Пример признаков
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features)
    return kmeans

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        genre = request.form.get('genre', '').strip()
        year = request.form.get('year', '').strip()
        criticscore = request.form.get('criticscore', '').strip()

        query = "SELECT * FROM games WHERE 1=1"
        params = []

        if title:
            query += " AND title LIKE ?"
            params.append(f'%{title}%')
        if genre:
            query += " AND genre LIKE ?"
            params.append(f'%{genre}%')
        if year:
            query += " AND year = ?"
            params.append(year)
        if criticscore:
            query += " AND criticscore >= ?"
            params.append(criticscore)

        conn = get_db_connection()
        all_games = conn.execute(query, params).fetchall()
        random_games = random.sample(all_games, min(5, len(all_games)))
        conn.close()

        excluded_ids = [game['id'] for game in random_games]
        return render_template('search.html', games=random_games, genre=genre, offset=5, excluded_ids=excluded_ids)

    return render_template('search.html', games=[])

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    if request.method == 'POST':
        title = request.form.get('title', '').strip()

        conn = get_db_connection()
        if title:
            game = conn.execute('SELECT * FROM games WHERE title LIKE ?', (f'%{title}%',)).fetchone()
            if game:
                genre = game['genre']
                recommendations = conn.execute('SELECT * FROM games WHERE genre = ? AND title != ? LIMIT 5', (genre, title)).fetchall()
            else:
                recommendations = []
        else:
            played_games = conn.execute('SELECT game_id FROM played_games ORDER BY RANDOM() LIMIT 2').fetchall()
            recommendations = []
            for played_game in played_games:
                game = conn.execute('SELECT * FROM games WHERE id = ?', (played_game['game_id'],)).fetchone()
                if game:
                    recommendations.append(game)
        conn.close()

        # Используем обученную модель для улучшения рекомендаций
        try:
            model = train_model(get_data())
            improved_recommendations = improve_recommendations(model, recommendations)
        except ValueError as e:
            print(e)
            improved_recommendations = recommendations

        return render_template('recommendations.html', recommendations=improved_recommendations)

    return render_template('recommendations.html', recommendations=[])

def improve_recommendations(model, recommendations):
    improved_recommendations = []
    for game in recommendations:
        features = pd.DataFrame([[game['year'], game['criticscore']]], columns=['year', 'criticscore'])
        cluster = model.predict(features)[0]
        game['cluster'] = cluster
        improved_recommendations.append(game)
    return improved_recommendations

@app.route('/mark_played', methods=['POST'])
def mark_played():
    data = json.loads(request.data)
    game_id = data['game_id']
    genre = data['genre']
    offset = data['offset']
    excluded_ids = data['excluded_ids']

    conn = get_db_connection()
    game = conn.execute('SELECT id, title, year, genre, criticscore FROM games WHERE id = ?', (game_id,)).fetchone()
    if game:
        conn.execute('INSERT INTO played_games (game_id, title, year, genre, criticscore) VALUES (?, ?, ?, ?, ?)',
                     (game['id'], game['title'], game['year'], game['genre'], game['criticscore']))
        conn.commit()
    else:
        conn.close()
        return jsonify({'message': 'Game not found'}), 404

    excluded_ids.append(game_id)

    query = "SELECT * FROM games WHERE genre = ? AND id NOT IN ({}) LIMIT 1".format(','.join('?' * len(excluded_ids)))
    params = [genre, *excluded_ids]

    new_game = conn.execute(query, params).fetchone()
    conn.close()

    if new_game:
        return jsonify({
            'id': new_game['id'],
            'title': new_game['title'],
            'genre': new_game['genre'],
            'year': new_game['year'],
            'developer': new_game['developer'],
            'publisher': new_game['publisher'],
            'platform': new_game['platform'],
            'criticscore': new_game['criticscore'],
            'userscore': new_game['userscore'],
            'poster': new_game['poster']
        })
    else:
        return jsonify({'message': 'No more games available'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
