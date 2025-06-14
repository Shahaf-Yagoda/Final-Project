-- Seed Data for Exercise Table
-- Date: 2025-06-14
-- Description: Common exercises supported by the fitness tracking system

BEGIN;

-- Insert common exercises with detailed information
INSERT INTO Exercise (name, description, target_muscles, instructions) VALUES
('lunge', 
 'Forward lunge exercise for lower body strength and stability', 
 '["quadriceps", "glutes", "hamstrings", "calves", "core"]'::jsonb, 
 'Step forward with one leg, lowering your hips until both knees are bent at about 90 degrees. Make sure your front knee is directly above your ankle, not pushed out too far. Keep your core engaged and your back straight. Push through your front heel to return to the starting position.'),

('press', 
 'Overhead press exercise for upper body strength', 
 '["shoulders", "triceps", "upper_chest", "core"]'::jsonb, 
 'Start with weights at shoulder level. Press the weights straight up overhead while keeping your core tight and spine neutral. Don''t arch your back excessively. Lower the weights back to shoulder level with control. Keep your elbows slightly forward, not flared out completely to the sides.'),

('plank', 
 'Plank hold exercise for core stability and strength', 
 '["core", "shoulders", "glutes", "back"]'::jsonb, 
 'Start in a push-up position but rest on your forearms instead of your hands. Keep your body in a straight line from head to heels. Engage your core muscles and avoid letting your hips sag or pike up. Hold this position while breathing normally. Focus on maintaining proper alignment throughout the hold.'),

('push_up', 
 'Classic push-up exercise for upper body and core strength', 
 '["chest", "shoulders", "triceps", "core"]'::jsonb, 
 'Start in a plank position with hands slightly wider than shoulder-width apart. Lower your chest toward the floor by bending your elbows, keeping them at about a 45-degree angle from your body. Push back up to the starting position. Maintain a straight line from head to heels throughout the movement.'),

('squat', 
 'Bodyweight squat for lower body strength and mobility', 
 '["quadriceps", "glutes", "hamstrings", "calves", "core"]'::jsonb, 
 'Stand with feet shoulder-width apart, toes slightly pointed out. Lower your body by pushing your hips back and bending your knees as if sitting in a chair. Keep your chest up and weight on your heels. Lower until your thighs are parallel to the ground, then push through your heels to return to standing.'),

('deadlift', 
 'Deadlift movement pattern for posterior chain strength', 
 '["hamstrings", "glutes", "lower_back", "traps", "rhomboids", "core"]'::jsonb, 
 'Stand with feet hip-width apart, weights in front of your legs. Hinge at the hips by pushing your hips back while keeping a slight bend in your knees. Lower the weights while maintaining a straight back and neutral spine. Drive through your heels and push your hips forward to return to standing.'),

('bicep_curl', 
 'Bicep curl exercise for arm strength', 
 '["biceps", "forearms"]'::jsonb, 
 'Stand with weights in both hands, arms at your sides, palms facing forward. Keep your elbows close to your sides and curl the weights up toward your shoulders by contracting your biceps. Slowly lower the weights back to the starting position with control.'),

('mountain_climber', 
 'Dynamic mountain climber exercise for cardio and core', 
 '["core", "shoulders", "hip_flexors", "quadriceps"]'::jsonb, 
 'Start in a high plank position. Bring one knee toward your chest, then quickly switch legs as if you''re running in place while holding the plank position. Keep your core engaged and maintain a straight line with your body. Alternate legs at a controlled pace.'),

('burpee', 
 'Full-body burpee exercise for conditioning', 
 '["chest", "shoulders", "triceps", "core", "quadriceps", "glutes", "hamstrings"]'::jsonb, 
 'Start standing, then squat down and place your hands on the floor. Jump your feet back into a plank position, perform a push-up (optional), then jump your feet back to the squat position. Explosively jump up with your arms overhead. Land softly and immediately begin the next repetition.'),

('side_plank', 
 'Side plank exercise for lateral core stability', 
 '["obliques", "core", "shoulders", "glutes"]'::jsonb, 
 'Lie on your side with your forearm on the ground, elbow directly under your shoulder. Stack your feet or place the top foot in front for more stability. Lift your hips off the ground, creating a straight line from head to feet. Hold this position while breathing normally. Focus on not letting your hips sag or rotate.')

ON CONFLICT (name) DO UPDATE SET
    description = EXCLUDED.description,
    target_muscles = EXCLUDED.target_muscles,
    instructions = EXCLUDED.instructions,
    updated_at = CURRENT_TIMESTAMP;

COMMIT;