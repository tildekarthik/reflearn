# SQL tutorial from SQLBolt for reference
Original work for reference is : https://sqlbolt.com/

This is intended as my cheatsheet for quick references for me to cut past - kindof like my notebook

## Simple query
`SELECT * 
FROM mytable;`

## Query with constraints
`SELECT column, another_column, …
FROM mytable
WHERE condition
    AND/OR another_condition
    AND/OR …;`

Criteria can be
### Numeric Criteria
- Operators (>,<,!=)
- BETWEEN ... AND ...
- NOT BETWEEN ... AND ...
- IN (li1, li2,...,li N)
### Text criteria
- Operators (= , != or <>)
- `LIKE` "%Str_to_match%" - Note the wildcard `%`
- `_` use to denote a single character
### NULL
`WHERE col_1 IS/IS NOT NULL`

### Expressions
`SELECT particle_speed / 2.0 AS half_particle_speed
FROM physics_data
WHERE ABS(particle_position) * 10.0 > 500;`
- Both column names and table names can be set with AS Eg. 
`SELECT column AS better_column_name, …
FROM a_long_widgets_table_name AS mywidgets
INNER JOIN widget_sales
  ON mywidgets.id = widget_sales.widget_id;`


## Filter and sort
### Distinct rows - drop duplicates
`SELECT DISTINCT column, another_column, …
FROM mytable
WHERE condition(s);`

### Sort order
`SELECT column, another_column, …
FROM mytable
WHERE condition(s)
ORDER BY column ASC/DESC;`

### Limit the responses
`SELECT column, another_column, …
FROM mytable
WHERE condition(s)
ORDER BY column ASC/DESC
LIMIT num_limit OFFSET num_offset;`

## Multi table queries
### JOIN
`SELECT column, another_table_column, …
FROM mytable
INNER JOIN another_table 
    ON mytable.id = another_table.id
WHERE condition(s)
ORDER BY column, … ASC/DESC
LIMIT num_limit OFFSET num_offset;`

- Default is INNER JOIN (just JOIN)
- Other joins LEFT/RIGHT/FULL JOIN

## Aggregation
### Single value return - Without grouping
`SELECT AGG_FUNC(column_or_expression) AS aggregate_description, …
FROM mytable
WHERE constraint_expression;`

### Summary Table return - With grouping
`SELECT group_by_column, AGG_FUNC(column_expression) AS aggregate_result_alias, …
FROM mytable
WHERE condition
GROUP BY column
HAVING group_condition;`

Note the `HAVING` used for filtering `GROUP BY` rows from result

#### Agg functions
- `COUNT(col - optional)`
- MIN / MAX / AVG / SUM (column)

## Subqueries and correlated sub_queries
`SELECT *
FROM sales_associates
WHERE salary > 
   (SELECT AVG(revenue_generated)
    FROM sales_associates);`
Correlated query:
`SELECT *
FROM employees
WHERE salary > 
   (SELECT AVG(revenue_generated)
    FROM employees AS dept_employees
    WHERE dept_employees.department = employees.department);`

## Unions, Intersections and Exceptions
`SELECT column, another_column
   FROM mytable
UNION / UNION ALL / INTERSECT / EXCEPT
SELECT other_column, yet_another_column
   FROM another_table
ORDER BY column DESC
LIMIT n;`

(Please see thhe order of execution section below for reference - UNION is before ORDER BY and LIMIT)

## Order of execution
Take the full expression below:
`SELECT DISTINCT column, AGG_FUNC(column_or_expression), …
FROM mytable
    JOIN another_table
      ON mytable.column = another_table.column
    WHERE constraint_expression
    GROUP BY column
    HAVING constraint_expression
    ORDER BY column ASC/DESC
    LIMIT count OFFSET COUNT;`
- The order of execution is: (1) FROM JOIN (2) WHERE (3) GROUPBY (4) HAVING (5) SELECT (6) DISTINCT (7) ORDER BY (8) LIMIT / OFFSET


## Insert or add rows to an existing table
`INSERT INTO mytable
(column, another_column, …)
VALUES (value_or_expr, another_value_or_expr, …),
      (value_or_expr_2, another_value_or_expr_2, …),
      …;`

The columns can be left out if all the data is available

## Update existing rows - using the constraints
`UPDATE mytable
SET column = value_or_expr, 
    other_column = another_value_or_expr, 
WHERE condition;`

! Check the where statement with a query before applying

## Deleting rows
`DELETE FROM mytable
WHERE condition;`
! Check the where condition

## Schema level operations - TABLE modifiers
### Creat TABLE

`CREATE TABLE IF NOT EXISTS mytable (
    column DataType TableConstraint DEFAULT default_value,
    another_column DataType TableConstraint DEFAULT default_value,
    …
);`

- Datatypes: INTEGER, BOOLEAN (also 0 or 1 only), FLOAT, DOUBLE, REAL,CHARACTER(num chars), VARCHAR(num char), TEXT, DATE, DATETIME, BLOB
- Table constraints: PRIMARY KEY, AUTOINCREMENT, UNIQUE, NOT NULL, CHECK(expression),FOREIGN KEY

### Altering tables
#### Adding columns
`ALTER TABLE mytable
ADD column DataType OptionalTableConstraint 
    DEFAULT default_value;`

#### Removing columns
`ALTER TABLE mytable
DROP column_to_be_deleted;`

#### Renaming a table
`ALTER TABLE mytable
RENAME TO new_table_name;`

### Dropping tables
`DROP TABLE IF EXISTS mytable;`

