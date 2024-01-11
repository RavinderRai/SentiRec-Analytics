USE SentiRec_Analytics;

-- Want to change name of headphone name columns to all be consistent.
-- Replace 'YourTable' with the actual table name and 'OldColumnName' with the actual column name
EXEC sp_rename 'amazon_product_descriptions.Headphones', 'headphoneName', 'COLUMN';


USE SentiRec_Analytics;
-- keep naming convention consistent
EXEC sp_rename 'amazon_reviews_dim_table.PrimaryIndex', 'primaryKey', 'COLUMN';

USE SentiRec_Analytics;
EXEC sp_rename 'averaged_embeddings.Headphone_Name', 'headphoneName', 'COLUMN';

USE SentiRec_Analytics;
-- keep naming convention consistent
EXEC sp_rename 'headphones_fact_table.primary_key', 'primaryKey', 'COLUMN';

USE SentiRec_Analytics;
-- keep naming convention consistent
EXEC sp_rename 'yt_reviews_gen_summaries.primary_key', 'primaryKey', 'COLUMN';
EXEC sp_rename 'yt_reviews_gen_summaries.headphone', 'headphoneName', 'COLUMN';


-- making headphoneName a foreign key
ALTER TABLE headphones_fact_table
ADD CONSTRAINT FK_HeadphoneName_Averaged_Embeddings FOREIGN KEY (headphoneName) REFERENCES averaged_embeddings(headphoneName);

SELECT Distinct(headphoneName)
FROM headphones_fact_table
ORDER BY headphoneName DESC;

SELECT Distinct(headphoneName)
FROM averaged_embeddings
ORDER BY headphoneName DESC;

-- slight error with a specific value that needs fixing
UPDATE averaged_embeddings
SET headphoneName = 'lg tone tf8'
WHERE headphoneName = 'Buy LG TONE TF8';


ALTER TABLE headphones_fact_table
ADD CONSTRAINT FK_HeadphoneName_amazon_product_descriptions FOREIGN KEY (headphoneName) REFERENCES amazon_product_descriptions(headphoneName);

-- foreign keys link to primary keys, but these have generic primary keys so can't link to them, but we don't really need to
-- but if when redoing the data collection phase with better APIs later, remember to add a special key for these tables to link
ALTER TABLE headphones_fact_table
ADD CONSTRAINT FK_HeadphoneName_yt_reviews_gen_summaries FOREIGN KEY (headphoneName) REFERENCES yt_reviews_gen_summaries(headphoneName);

ALTER TABLE headphones_fact_table
ADD CONSTRAINT FK_HeadphoneName_amazon_reviews_dim_table FOREIGN KEY (headphoneName) REFERENCES amazon_reviews_dim_table(headphoneName);