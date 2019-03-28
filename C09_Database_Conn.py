import sqlite3
import os

if os.path.exists('reviews.sqlite'):
    os.remove('reviews.sqlite')
conn=sqlite3.connect('movieclassifier/reviews.sqlite')
c=conn.cursor()

# c.execute('Create table review_db (review text,sentiment integer,date text)')
# ex1="I love this movie"
# c.execute("insert into review_db (review,sentiment,date) values (?,?,DATETIME('now'))",(ex1,1))
# ex2="I dislike this movie"
# c.execute("insert into review_db values (?,?,DATETIME('now'))",(ex2,0))

c.execute("Select * from review_db")
result=c.fetchall()

# conn.commit()
conn.close()

print(result)