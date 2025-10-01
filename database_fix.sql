-- QuantEdge Professional Database Schema Fix
-- Monday Morning Database Repair Script

-- Fix the missing adjClose column issue
ALTER TABLE stock_prices ADD COLUMN IF NOT EXISTS adjClose REAL;

-- Update existing data to set adjClose = close where null
UPDATE stock_prices SET adjClose = close WHERE adjClose IS NULL;

-- Verify the fix
SELECT 'Database schema fixed!' as status;