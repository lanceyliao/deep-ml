WITH monthly_valid_purchases AS (
    -- 第一层：筛选年份，并按“用户”和“月份”统计达标月份
    SELECT 
        user_id,
        EXTRACT(MONTH FROM purchase_date) AS purchase_month
    FROM purchases
    WHERE purchase_date >= DATE '2024-01-01' 
      AND purchase_date < DATE '2025-01-01'
    GROUP BY 
        user_id, 
        EXTRACT(MONTH FROM purchase_date)
    HAVING COUNT(purchase_id) >= 2
)
-- 第二层：统计每个用户“达标的月份数”，判断是否全勤（等于 12 个月）
SELECT 
    user_id
FROM monthly_valid_purchases
GROUP BY user_id
HAVING COUNT(purchase_month) = 12
ORDER BY user_id ASC;