import sqlalchemy
from sqlalchemy import select
from sqlalchemy.orm import Session
import numpy as np
from database import engine, Images
# reusable function to insert data into the table
def insert_image(engine: sqlalchemy.Engine, image_path: str, image_embedding: list[float]):
    with Session(engine) as session:
        # create the image object
        image = Images(
            image_path=image_path,
            image_embedding=image_embedding,
        )
        # add the image object to the session
        session.add(image)
        # commit the transaction
        session.commit()

# insert some data into the table
N = 100
for i in range(N):
    image_path = f"image_{i}.jpg"
    image_embedding = np.random.rand(512).tolist()
    insert_image(engine, image_path, image_embedding)

# select first image from the table
with Session(engine) as session:
    image = session.query(Images).first()


# calculate the cosine similarity between the first image and the K rest of the images, order the images by the similarity score
def find_k_images(engine: sqlalchemy.Engine, k: int, orginal_image: Images) -> list[Images]:
    with Session(engine) as session:
        # execution_options={"prebuffer_rows": True} is used to prebuffer the rows, this is useful when we want to fetch the rows in chunks and return them after session is closed
        result = session.execute(
            select(Images)
            .order_by(Images.image_embedding.cosine_distance(orginal_image.image_embedding))
            .limit(k), 
            execution_options={"prebuffer_rows": True}
        )
        #return result
        return list(result.scalars().all())

# find the images with the similarity score greater than 0.9
def find_images_with_similarity_score_greater_than(engine: sqlalchemy.Engine, similarity_score: float, orginal_image: Images) -> list[Images]:
    with Session(engine) as session:
        max_distance = 1 - similarity_score 
        result = session.execute(
            select(Images)
            .filter(Images.image_embedding.cosine_distance(orginal_image.image_embedding) < max_distance), 
            execution_options={"prebuffer_rows": True}
        )
        #return result
        return list(result.scalars().all())



# find the 10 most similar images to the first image
k = 10
similar_images = find_k_images(engine, k, image)
print("The most similarity images:")
for img in similar_images:
    print(img.id, img.image_path)



min_similarity = 0.9
similar_enough = find_images_with_similarity_score_greater_than(
    engine,
    min_similarity,
    image,
)

print(f"\nImage with similarity > {min_similarity}:")
for img in similar_enough:
    print(img.id, img.image_path)
