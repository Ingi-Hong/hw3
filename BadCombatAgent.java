package hw3;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.State.StateView;

public class BadCombatAgent extends Agent {
	
	private int enemyPlayerNum = 1;

	public BadCombatAgent(int playernum, String[] args)
	{
		super(playernum);
		
		System.out.println("Constructed MyCombatAgent");
	}

	@Override
	public Map<Integer, Action> initialStep(StateView state, HistoryView history)
	{
		// figure out who the enemy is
		Set<Integer> playerIds = new HashSet<Integer>();
		for(Integer playerId : state.getPlayerNumbers())
		{
			playerIds.add(playerId);
		}
		if(playerIds.size() != 2)
		{
			System.err.println("BadCombatAgent.initialStep [ERROR]: only allowed to have 2 players, players=" + playerIds);
			System.exit(-1);
		}

		playerIds.remove(this.getPlayerNumber());
		this.enemyPlayerNum = playerIds.iterator().next();

		// This stores the action that each unit will perform
		// if there are no changes to the current actions then this
		// map will be empty
		Map<Integer, Action> actions = new HashMap<Integer, Action>();
		
		// This is a list of all of your units
		// Refer to the resource agent example for ways of
		// differentiating between different unit types based on
		// the list of IDs
		List<Integer> myUnitIDs = state.getUnitIds(playernum);
		
		// This is a list of enemy units
		List<Integer> enemyUnitIDs = state.getUnitIds(enemyPlayerNum);
		
		if(enemyUnitIDs.size() == 0)
		{
			// Nothing to do because there is no one left to attack
			return actions;
		}
		
		// start by commanding every single unit to attack an enemy unit
		for(Integer myUnitID : myUnitIDs)
		{
			// Command all of my units to attack the first enemy unit in the list
			actions.put(myUnitID, Action.createCompoundAttack(myUnitID, enemyUnitIDs.get(0)));
		}
		
		return actions;
	}

	@Override
	public Map<Integer, Action> middleStep(StateView state, HistoryView history) {
		// This stores the action that each unit will perform
		// if there are no changes to the current actions then this
		// map will be empty
		Map<Integer, Action> actions = new HashMap<Integer, Action>();
		
		// This is a list of enemy units
		List<Integer> enemyUnitIDs = state.getUnitIds(enemyPlayerNum);
		
		if(enemyUnitIDs.size() == 0)
		{
			// Nothing to do because there is no one left to attack
			return actions;
		}
		
		int currentStep = state.getTurnNumber();
		
		// go through the action history
		for(ActionResult feedback : history.getCommandFeedback(playernum, currentStep-1).values())
		{
			// if the previous action is no longer in progress (either due to failure or completion)
			// then add a new action for this unit
			if(feedback.getFeedback() != ActionFeedback.INCOMPLETE)
			{
				// attack the first enemy unit in the list
				int unitID = feedback.getAction().getUnitId();
				actions.put(unitID, Action.createCompoundAttack(unitID, enemyUnitIDs.get(0)));			
			}
		}

		return actions;
	}

	@Override
	public void terminalStep(StateView newstate, HistoryView statehistory) {
		System.out.println("Finished the episode");
	}

	@Override
	public void savePlayerData(OutputStream os) {
		// TODO Auto-generated method stub

	}

	@Override
	public void loadPlayerData(InputStream is) {
		// TODO Auto-generated method stub

	}

}
