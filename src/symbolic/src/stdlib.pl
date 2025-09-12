% Standard library for symbolic reasoning engine
% Core Prolog predicates for compliance reasoning

% Meta-predicates
:- op(700, xfy, '->').
:- op(1050, xfy, '->').
:- op(1100, xfy, ';').

% Basic list operations
member(X, [X|_]).
member(X, [_|T]) :- member(X, T).

append([], L, L).
append([H|T], L, [H|R]) :- append(T, L, R).

length([], 0).
length([_|T], N) :- length(T, M), N is M + 1.

% Compliance reasoning predicates
compliant_system(System) :- 
    has_encryption(System),
    has_access_control(System),
    has_monitoring(System).

security_gap(System, Control) :-
    required_control(System, Control),
    \+ implemented_control(System, Control).

audit_required(System) :-
    compliance_framework(Framework),
    bound_to_framework(System, Framework),
    audit_interval_exceeded(System, Framework).

% Data classification helpers  
classify_data(Data, sensitive) :- contains_pii(Data).
classify_data(Data, sensitive) :- contains_payment_info(Data).
classify_data(Data, public) :- \+ classify_data(Data, sensitive).

% Risk assessment
risk_level(System, high) :-
    processes_sensitive_data(System),
    \+ compliant_system(System).

risk_level(System, medium) :-
    processes_sensitive_data(System),
    compliant_system(System).

risk_level(System, low) :-
    \+ processes_sensitive_data(System).

% Temporal reasoning
within_timeframe(Event, Start, End) :-
    event_timestamp(Event, Time),
    Time >= Start,
    Time =< End.

overdue(Task, CurrentTime) :-
    task_deadline(Task, Deadline),
    CurrentTime > Deadline.