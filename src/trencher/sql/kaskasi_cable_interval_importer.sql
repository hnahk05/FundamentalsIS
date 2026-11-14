SELECT "Cycle", MIN("Start UTC") "Start UTC", MAX("End UTC") "End UTC", MAX("End UTC") - MIN("Start UTC") "Duration" 
FROM time_registration_v2 
WHERE "Project" LIKE '5972%' AND "Subactivity" = 'Trenching Pass 1'
GROUP BY "Cycle"
ORDER BY "Start UTC"