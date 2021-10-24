## Inspiration
Our team wanted to address the challenge #01 presented by EduHack: "Support for collective musical creation.
The project is part of MusicEduHack's expressed need to generate a tool that, through AI techniques, helps collective composition.
## What it does
It is proposed as a tool to use in two areas:
1.- Collective live performance, for example a DJ assisted by an AI that collects parameters of the movements of the mobile phone of the public and these, combined, generate a MIDI melody that will serve as material to generate music.
The DJ is no longer the only one who is generating everything that happens musically (or visually) speaking, but the public is participating by providing sound material or producing some kind of variation (for example filters) on what the DJ produces. Thus diluting the barrier between the artist (star) and the public.
2.-The other possible scenario is an improvisation with two or three people, in which the MIDI musical data generated by the performers generate an automatic melody, producing a dialogue (a very common resource or strategy in musical improvisation), producing a dialogue between performers and a AI previously trained.
This project has focused on generating a melody from several melodies (sequences of notes) using the magenta RNN library.
## How to use
The conda environment is contained in the environment.yml file. To train the model, we need to download cat-mel_2bar_big.tar from the magenta website and place it in the /contained folder.
The main part of the project is in the notebook "MusicAverage" and in the patch Data-To-Midi.maxpat.
## How we built it
Using Cycling '74 Max, Python and Magenta Project. Sensors2OSC.
## Challenges we ran into.
Understanding how Magenta Project is working
## Accomplishments that we're proud of
Generating
## What we learned.
Magenta project possibilities
## What's next for Untitled