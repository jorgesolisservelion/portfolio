USE Wedding_database;

#We are browsing in the database
SELECT p.product_id, p.product_name, v.vendor_id, p.price_unit, p.unit_vol, p.price_ce, 
v.vendor_depart, v.vendor_name, v.vendor_location, v.vendor_sustainable, 
CASE 
	WHEN v.vendor_rating = 0 THEN ''
    ELSE v.vendor_rating
END AS rating_vendor,
fs.flower_season, fs.flower_style, vs.artificial_flowers_in_portfolio,
att.color, att.tie, att.number_of_buttons, att.lapel_style,
cat.category_name,
sus.equip_ec, sus.avg_usage_hours, sus.total_ec, sus.number_equipment
FROM Products as p -- Main table
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
LEFT JOIN Flower_Season_Style as fs
ON p.product_id = fs.product_id
LEFT JOIN Flowers_Vendor_Sustainability as vs -- Join with subtable
ON v.vendor_id = vs.vendor_id
LEFT JOIN attire as att
ON p.product_id = att.product_id
LEFT JOIN categories as cat -- Join with anothe subtable
ON p.product_id = cat.product_id
LEFT JOIN Sustainability as sus
ON v.vendor_id = sus.vendor_id
;

#Obtaining the number of records by department
SELECT vendor_depart, COUNT(*) AS obs, ROUND((COUNT(*)*100/(SELECT COUNT(*) FROM Products)),2)AS perc
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY vendor_depart
ORDER BY obs desc;

#Analyzing the price unit
SELECT p.unit_vol, count(*) as obs
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY p.unit_vol
ORDER BY obs desc;

#We can categorize the units of price measurement
SELECT CASE
	WHEN p.unit_vol IN ('per person','1 per bride side','1 per bride trial','1 per groom side','1 per kids','1 per bride','1 per person') THEN 'Individual Services'
    WHEN p.unit_vol IN ('per service','1 per vendor','1','unit') THEN 'Per Unit Services'
    WHEN p.unit_vol IN ('per 100 invitations','6 hours','1 per table','1 per piece','1 per chair','1 per traditional','1 per backdrop','1 per airbrush') THEN 'Event Supplies'
    ELSE 'Others'
    END AS unit_v, count(*)
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY unit_v;

#Now lets see location
SELECT v.vendor_location, count(*) as obs
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY v.vendor_location
ORDER BY obs desc; #We can see that we have several observations with the same name but with different capitalizations

#Lets normalize the location data
SELECT CASE
        WHEN v.vendor_location IN ('san francisco', 'san francisco ', ' san francisco ') THEN 'San Francisco'
        WHEN v.vendor_location IN ('san jose', 'san jose ') THEN 'San Jose'
        WHEN v.vendor_location IN ('oakland', 'oakland ') THEN 'Oakland'
        WHEN v.vendor_location IN ('santa clara', 'santa clara ') THEN 'Santa Clara'
        WHEN v.vendor_location IN ('berkeley', 'berkeley ') THEN 'Berkeley'
        WHEN v.vendor_location IN ('hayward', 'hayward ') THEN 'Hayward'
        WHEN v.vendor_location IN ('los gatos', 'los gatos ') THEN 'Los Gatos'
        WHEN v.vendor_location IN ('livermore ', 'livermore') THEN 'Livermore'
        WHEN v.vendor_location IN ('walnut creek ', 'walnut creek') THEN 'Walnut Creek'
        WHEN v.vendor_location IN ('sausolito', 'sausalito ','sausalito') THEN 'Sausalito'
        WHEN v.vendor_location IN ('san anselmo ', 'san alselmo') THEN 'San Alselmo'
        WHEN v.vendor_location IN ('redwood city', 'redwood') THEN 'Redwood'
        WHEN v.vendor_location IN ('fremont', 'freemont ') THEN 'Freemont'
        WHEN v.vendor_location IN ('concord ', 'concord',' concord ') THEN 'Concord'
        ELSE v.vendor_location
    END AS standardized_location,
    count(*) as obs
FROM Products as p -- We got a standard data of location
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY standardized_location
ORDER BY standardized_location desc;

#We can group the locations in areas

SELECT
    CASE
        WHEN standardized_location IN ('San Francisco', 'Oakland', 'Berkeley', 'Hayward', 'Livermore', 
                                       'Los Gatos', 'Walnut Creek', 'Concord', 'castro valley', 
                                       'san rafael', 'novato ', 'napa', 'brisbane', 
                                       'san luis obispo', 'San Alselmo', 'sacramento', 'daly city', 
                                       'morro bay', 'monterey', 'milbrae ', 'martinez', 
                                       'greenbrae', 'el cerrito ', 'alameda', 'palo alto', 
                                       'burlingame', 'sunnyvale', 'hillsborough', 'lafayette', 
                                       'san leandro', 'san carlos', 'gilroy', 'san mateo ', 
                                       'south san francisco', 'scotts valley', 'paso robles', 'petaluma', 
                                       'watsonville', 'vacaville ', 'tiburon ', 'sunol ', 'studio', 
                                       'stanford ', 'san ramon ', 'san joaquin valley', 'san diego', -- Grouping locations
                                       'saint martin', 'richmond ', 'pleasanton', 'pleasant hill', 
                                       'pittsburg', 'pescadero ', 'oakley ', 'oakley', 'nicasio ', 
                                       'menlo park ', 'mammoth lakes ', 'hollister ', 'hercules', 
                                       'felton ', 'felton', 'dublin ', 'dixon ', 'cupertino ', 
                                       'cupertino', 'corte madera', 'cloverdale ', 'clayton ', 
                                       'carmel', 'campbell ', 'campbell', 'calistoga ', 'brentwood ', 
                                       'belmont', 'antioch', 'acampo ') THEN 'San Francisco Bay Area'
        WHEN standardized_location IN ('San Jose', 'Santa Clara') THEN 'South Bay'
        WHEN standardized_location IN ('Sausalito', 'Tiburon') THEN 'North Bay'
        WHEN standardized_location IN ('Fremont', 'Livermore', 'Pleasanton ', 'mountain view ', 'half moon bay ') THEN 'East Bay'
        WHEN standardized_location IN ('palo alto', 'Redwood', 'Menlo Park') THEN 'Peninsula'
        WHEN standardized_location = 'online' THEN 'Online' -- We group locations
        ELSE 'Other'
    END AS region_group,
    count(*) as obs
FROM (SELECT CASE -- We need to create a subquery 
        WHEN v.vendor_location IN ('san francisco', 'san francisco ', ' san francisco ') THEN 'San Francisco'
        WHEN v.vendor_location IN ('san jose', 'san jose ') THEN 'San Jose'
        WHEN v.vendor_location IN ('oakland', 'oakland ') THEN 'Oakland'
        WHEN v.vendor_location IN ('santa clara', 'santa clara ') THEN 'Santa Clara'
        WHEN v.vendor_location IN ('berkeley', 'berkeley ') THEN 'Berkeley'
        WHEN v.vendor_location IN ('hayward', 'hayward ') THEN 'Hayward' 
        WHEN v.vendor_location IN ('los gatos', 'los gatos ') THEN 'Los Gatos'
        WHEN v.vendor_location IN ('livermore ', 'livermore') THEN 'Livermore'
        WHEN v.vendor_location IN ('walnut creek ', 'walnut creek') THEN 'Walnut Creek'
        WHEN v.vendor_location IN ('sausolito', 'sausalito ','sausalito') THEN 'Sausalito'
        WHEN v.vendor_location IN ('san anselmo ', 'san alselmo') THEN 'San Alselmo'
        WHEN v.vendor_location IN ('redwood city', 'redwood') THEN 'Redwood'
        WHEN v.vendor_location IN ('fremont', 'freemont ') THEN 'Freemont'
        WHEN v.vendor_location IN ('concord ', 'concord',' concord ') THEN 'Concord'
        ELSE v.vendor_location
    END AS standardized_location
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id
GROUP BY standardized_location
ORDER BY standardized_location desc) as SUBQUERY -- this is the subquery
GROUP BY region_group
ORDER BY region_group desc;

#Join and create a table
SELECT p.product_id, p.product_name, v.vendor_id, p.price_unit, p.unit_vol, p.price_ce, 
v.vendor_depart, v.vendor_name, v.vendor_location, 
CASE
        WHEN v.vendor_location IN ('san francisco', 'san francisco ', ' san francisco ') THEN 'San Francisco'
        WHEN v.vendor_location IN ('san jose', 'san jose ') THEN 'San Jose'
        WHEN v.vendor_location IN ('oakland', 'oakland ') THEN 'Oakland'
        WHEN v.vendor_location IN ('santa clara', 'santa clara ') THEN 'Santa Clara'
        WHEN v.vendor_location IN ('berkeley', 'berkeley ') THEN 'Berkeley' -- Our first case
        WHEN v.vendor_location IN ('hayward', 'hayward ') THEN 'Hayward'
        WHEN v.vendor_location IN ('los gatos', 'los gatos ') THEN 'Los Gatos'
        WHEN v.vendor_location IN ('livermore ', 'livermore') THEN 'Livermore'
        WHEN v.vendor_location IN ('walnut creek ', 'walnut creek') THEN 'Walnut Creek'
        WHEN v.vendor_location IN ('sausolito', 'sausalito ','sausalito') THEN 'Sausalito'
        WHEN v.vendor_location IN ('san anselmo ', 'san alselmo') THEN 'San Alselmo'
        WHEN v.vendor_location IN ('redwood city', 'redwood') THEN 'Redwood'
        WHEN v.vendor_location IN ('fremont', 'freemont ') THEN 'Freemont'
        WHEN v.vendor_location IN ('concord ', 'concord',' concord ') THEN 'Concord'
        ELSE v.vendor_location
    END AS standardized_location,
CASE
        WHEN standardized_location IN ('San Francisco', 'Oakland', 'Berkeley', 'Hayward', 'Livermore', -- Our second Case
                                       'Los Gatos', 'Walnut Creek', 'Concord', 'castro valley', 
                                       'san rafael', 'novato ', 'napa', 'brisbane', 
                                       'san luis obispo', 'San Alselmo', 'sacramento', 'daly city', 
                                       'morro bay', 'monterey', 'milbrae ', 'martinez', 
                                       'greenbrae', 'el cerrito ', 'alameda', 'palo alto', 
                                       'burlingame', 'sunnyvale', 'hillsborough', 'lafayette', 
                                       'san leandro', 'san carlos', 'gilroy', 'san mateo ', 
                                       'south san francisco', 'scotts valley', 'paso robles', 'petaluma', 
                                       'watsonville', 'vacaville ', 'tiburon ', 'sunol ', 'studio', 
                                       'stanford ', 'san ramon ', 'san joaquin valley', 'san diego', 
                                       'saint martin', 'richmond ', 'pleasanton', 'pleasant hill', 
                                       'pittsburg', 'pescadero ', 'oakley ', 'oakley', 'nicasio ', 
                                       'menlo park ', 'mammoth lakes ', 'hollister ', 'hercules',
                                       'felton ', 'felton', 'dublin ', 'dixon ', 'cupertino ', 
                                       'cupertino', 'corte madera', 'cloverdale ', 'clayton ', 
                                       'carmel', 'campbell ', 'campbell', 'calistoga ', 'brentwood ', 
                                       'belmont', 'antioch', 'acampo ') THEN 'San Francisco Bay Area'
        WHEN standardized_location IN ('San Jose', 'Santa Clara') THEN 'South Bay'
        WHEN standardized_location IN ('Sausalito', 'Tiburon') THEN 'North Bay'
        WHEN standardized_location IN ('Fremont', 'Livermore', 'Pleasanton ', 'mountain view ', 'half moon bay ') THEN 'East Bay'
        WHEN standardized_location IN ('palo alto', 'Redwood', 'Menlo Park') THEN 'Peninsula'
        WHEN standardized_location = 'online' THEN 'Online'
        ELSE 'Other'
    END AS region_group,
v.vendor_sustainable, 
CASE -- Our thrird Case
	WHEN v.vendor_rating = 0 THEN ''
    ELSE v.vendor_rating
END AS rating_vendor,
CASE
	WHEN p.unit_vol IN ('per person','1 per bride side','1 per bride trial','1 per groom side','1 per kids','1 per bride','1 per person') THEN 'Individual Services'
    WHEN p.unit_vol IN ('per service','1 per vendor','1','unit') THEN 'Per Unit Services'
    WHEN p.unit_vol IN ('per 100 invitations','6 hours','1 per table','1 per piece','1 per chair','1 per traditional','1 per backdrop','1 per airbrush') THEN 'Event Supplies'
    ELSE 'Others'
    END AS unit_v, -- Our 4th Case
fs.flower_season, fs.flower_style, vs.artificial_flowers_in_portfolio,
att.color, att.tie, att.number_of_buttons, att.lapel_style,
cat.category_name,
sus.equip_ec, sus.avg_usage_hours, sus.total_ec, sus.number_equipment
FROM Products as p
INNER JOIN Vendors as v
ON p.vendor_id = v.vendor_id -- Joining all tables
LEFT JOIN Flower_Season_Style as fs
ON p.product_id = fs.product_id
LEFT JOIN Flowers_Vendor_Sustainability as vs
ON v.vendor_id = vs.vendor_id
LEFT JOIN attire as att
ON p.product_id = att.product_id
LEFT JOIN categories as cat
ON p.product_id = cat.product_id
LEFT JOIN Sustainability as sus
ON v.vendor_id = sus.vendor_id
LEFT JOIN ( -- Creating a subquery such as table source
SELECT vendor_id,
CASE
        WHEN v.vendor_location IN ('san francisco', 'san francisco ', ' san francisco ') THEN 'San Francisco'
        WHEN v.vendor_location IN ('san jose', 'san jose ') THEN 'San Jose'
        WHEN v.vendor_location IN ('oakland', 'oakland ') THEN 'Oakland'
        WHEN v.vendor_location IN ('santa clara', 'santa clara ') THEN 'Santa Clara'
        WHEN v.vendor_location IN ('berkeley', 'berkeley ') THEN 'Berkeley'
        WHEN v.vendor_location IN ('hayward', 'hayward ') THEN 'Hayward'
        WHEN v.vendor_location IN ('los gatos', 'los gatos ') THEN 'Los Gatos'
        WHEN v.vendor_location IN ('livermore ', 'livermore') THEN 'Livermore'
        WHEN v.vendor_location IN ('walnut creek ', 'walnut creek') THEN 'Walnut Creek'
        WHEN v.vendor_location IN ('sausolito', 'sausalito ','sausalito') THEN 'Sausalito'
        WHEN v.vendor_location IN ('san anselmo ', 'san alselmo') THEN 'San Alselmo'
        WHEN v.vendor_location IN ('redwood city', 'redwood') THEN 'Redwood'
        WHEN v.vendor_location IN ('fremont', 'freemont ') THEN 'Freemont'
        WHEN v.vendor_location IN ('concord ', 'concord',' concord ') THEN 'Concord'
        ELSE v.vendor_location
    END AS standardized_location
    FROM Vendors as V) AS sl
    ON v.vendor_id = sl.vendor_id; -- This is the table we are going to use for the analysis.





