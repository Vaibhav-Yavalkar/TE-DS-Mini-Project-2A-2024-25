const exercises = [ 
   ' Squats',
    'Deadlift',
    'Overhead Press',
    'Barbell Row',
    'Bicep Curls',
    'Tricep Dips',
    'Lunges',
    'Leg Press',
    'Running',
    'Cycling',
    'Jump Rope',
    'Push-Ups',
    'Pull-Ups',
    'Plank',
    'Mountain Climbers',
    'Rowing Machine',
    'Kettlebell Swings',
    'Step-Ups',
    'Battle Ropes',
    'Leg Curl',
    'Leg Extension',
    'Shoulder Press',
    'Triceps Kickback',
    'Upright Row',
    'Chest Flyes',
    'Hammer Curls',
    'Triceps Overhead Press',
    'Bulgarian Split Squats',
    'Lat Pulldown',
    
    'Seated Row',
    'Incline Bench Press',
    'Face Pulls',
    'Side Lateral Raise',
    'Sumo Deadlift',
    'Dumbbell Curl',
    'Skull Crushers',
    'Single-Leg Deadlift',
    'Bent-Over Row',
    'Dumbbell Lunges',
    'Front Raise',
    'Leg Raises',
    'Farmer Walk',
    'Cable Flyes',
    'Dumbbell Shrugs',
    'Box Jumps',
    'Treadmill Sprints',
    'Medicine Ball Slams',
    'Russian Twists',
    'Glute Bridge',
    'Cable Tricep Pushdown',
    'Reverse Fly',
    'Bent-Over Dumbbell Row',
    'Kettlebell Deadlift',
    'Dumbbell Flyes',
    'Leg Extension Machine',
    'Dumbbell Lateral Raise',
    'Incline Dumbbell Press',
    'Hip Thrust',
    'Cable Chest Flyes',
    'Standing Calf Raise',
    'Dumbbell Shoulder Press',
    'Seated Leg Curl',
    'Cable Crossovers',
    'Tricep Rope Pushdown',
    'Hanging Leg Raise',
    'Rope Face Pulls',
    'Step Mill',
    'Incline Treadmill Walk',
    'Medicine Ball Woodchops',
    'Reverse Lunge',
    'Chest Dips',
    'Landmine Press',
    'Hanging Knee Raise',
    'Dumbbell Front Raise',
    'Barbell Squats',
    'Dumbbell Bench Press',
    'Kettlebell Swings',
    'Dumbbell Kickback',
    'Seated Calf Raise',
    'Cable Lateral Raise',
    'Dumbbell Deadlift',
    'Renegade Rows',
    'Reverse Pec Deck Fly',
    'Dumbbell Side Bend',
    'Plank',
    'Push-Ups',
    'Walking Lunges',
    'Overhead Tricep Extension',
    'Single Arm Dumbbell Row'
   
];

document.getElementById('exercise-search').addEventListener('input', filterExercises);
document.getElementById('add-exercise-btn').addEventListener('click', showDropdown);
document.getElementById('delete-all-btn').addEventListener('click', deleteAllExercises);

let exerciseCount = 0;

// Load saved data on page load
window.onload = showData;

function filterExercises() {
    const searchInput = document.getElementById('exercise-search').value.toLowerCase();
    const filteredExercises = exercises.filter(exercise => exercise.toLowerCase().includes(searchInput));
    displayDropdown(filteredExercises);
}

function showDropdown() {
    const dropdown = document.getElementById('exercise-dropdown');
    dropdown.classList.remove('hidden');
    displayDropdown(exercises);
}

// Auto-close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const dropdown = document.getElementById('exercise-dropdown');
    const searchInput = document.getElementById('exercise-search');
    if (!dropdown.contains(event.target) && !searchInput.contains(event.target)) {
        dropdown.classList.add('hidden');
    }
});

function displayDropdown(exercisesArray) {
    const dropdown = document.getElementById('exercise-dropdown');
    dropdown.innerHTML = ''; // Clear previous dropdown items
    exercisesArray.forEach(exercise => {
        const li = document.createElement('li');
        li.textContent = exercise;
        li.addEventListener('click', () => {
            addExercise(exercise);
            document.getElementById('exercise-search').value = '';
            dropdown.classList.add('hidden');
        });
        dropdown.appendChild(li);
    });

    if (exercisesArray.length === 0) {
        dropdown.classList.add('hidden');
    } else {
        dropdown.classList.remove('hidden');
    }
}

// Function to add new exercise
function addExercise(exerciseName) {
    exerciseCount++;

    const exerciseDiv = document.createElement('div');
    exerciseDiv.classList.add('exercise');

    exerciseDiv.innerHTML = `
        <div class="exercise-header">
            <h3>${exerciseName}</h3>
            <button class="delete-exercise">Delete Exercise</button>
        </div>
        <div class="set-list">
            <div class="set">
                <input class="set-input" type="number" placeholder="Reps">
                <input class="set-input" type="number" placeholder="Kg">
                <button class="delete-set">X</button>
            </div>
        </div>
        <button class="add-set"> +Add Set</button>
    `;

    exerciseDiv.querySelector('.add-set').addEventListener('click', () => {
        addSet(exerciseDiv);
    });

    exerciseDiv.querySelector('.delete-exercise').addEventListener('click', () => {
        deleteExercise(exerciseDiv);
    });

    exerciseDiv.querySelector('.delete-set').addEventListener('click', (e) => {
        deleteSet(e.target.closest('.set'));
    });

    document.getElementById('exercise-list').appendChild(exerciseDiv);

    saveData();  // Save the data after adding exercise
}

function addSet(exerciseDiv) {
    const setList = exerciseDiv.querySelector('.set-list');

    const setDiv = document.createElement('div');
    setDiv.classList.add('set');
    setDiv.innerHTML = `
        <input class="set-input" type="number" placeholder="Reps">
        <input class="set-input" type="number" placeholder="Kg">
        <button class="delete-set">X</button>
    `;

    setList.appendChild(setDiv);

    setDiv.querySelector('.delete-set').addEventListener('click', (e) => {
        deleteSet(setDiv);
    });

    saveData();  // Save the data after adding set
}

function deleteSet(setDiv) {
    setDiv.remove();
    saveData();  // Save the data after deleting set
}

function deleteExercise(exerciseDiv) {
    exerciseDiv.remove();
    saveData();  // Save the data after deleting exercise
}

function deleteAllExercises() {
    document.getElementById('exercise-list').innerHTML = '';  // Clear the exercise list
    saveData();  // Save the data after deleting all exercises
}

// Save exercises to localStorage
function saveData() {
    const exerciseList = document.getElementById('exercise-list').innerHTML;
    localStorage.setItem('gymLoggerData', exerciseList);
}

// Load exercises from localStorage
function showData() {
    const savedData = localStorage.getItem('gymLoggerData');
    if (savedData) {
        document.getElementById('exercise-list').innerHTML = savedData;

        // Reattach event listeners after loading data
        document.querySelectorAll('.delete-exercise').forEach(button => {
            button.addEventListener('click', (e) => deleteExercise(e.target.closest('.exercise')));
        });

        document.querySelectorAll('.delete-set').forEach(button => {
            button.addEventListener('click', (e) => deleteSet(e.target.closest('.set')));
        });

        document.querySelectorAll('.add-set').forEach(button => {
            button.addEventListener('click', (e) => addSet(e.target.closest('.exercise')));
        });
    }
}
