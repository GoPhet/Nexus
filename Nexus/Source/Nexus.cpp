#include <math.h>
#include <CL\cl.h>
#include "Nexus.h"

using namespace Nexus;

inline int max(int a, int b)
{
	return a > b ? a : b;
}

inline int min(int a, int b)
{
	return a < b ? a : b;
}

inline double max(double a, double b)
{
	return a > b ? a : b;
}

inline double min(double a, double b)
{
	return a < b ? a : b;
}

Synaspe Synaspe::Random(Profile& profile)
{
	return Synaspe(Compute::Random(0, profile.neurons.size()-1), Compute::Random(), Compute::Random());
}

bool Neuron::Connected(unsigned int neuron, unsigned int* index)
{
	for (unsigned int x = 0; x < connections.size(); x++)
	{
		if (connections[x].connection == neuron) 
		{
			if (index != nullptr) *index = x;
			return true;
		}
	}
	return false;
}

double Neuron::Activate(double memory, double weight)
{
	return Compute::Sigmoid(bias + (memory * weight));
}

Neuron Neuron::Random(Profile& profile)
{
	unsigned int neurons = Compute::Random(1, profile.neurons.size()) + 1;
	std::vector<Synaspe> connections(neurons);
	for (unsigned int i = 0; i < neurons; i++) 
	{
		connections[i] = Synaspe::Random(profile);
	}
	return Neuron(Compute::Random(), connections);
}

bool Profile::Valid(unsigned int neuron)
{
	return neuron < neurons.size();
}

std::vector<Bridge> Profile::ConnectionsTo(unsigned int neuron)
{
	std::vector<Bridge> connnections;
	for (unsigned int i = 0; i < neurons.size(); i++)
	{
		for (unsigned int x = 0; x < neurons[i].connections.size(); x++)
		{
			if (neurons[i].connections[x].connection == neuron)
				connnections.push_back(Bridge(i, neuron, x));
		}
	}
	return connnections;
}

Network::Network(const Profile& profile)
{
	m_profile = profile;
	m_memory = std::vector<double>(profile.neurons.size());
}

void Nexus::Network::Wipe(unsigned int neuron)
{
	m_memory[neuron] = 0.0;
}

Profile Profile::Random(unsigned int maxNeurons)
{
	Profile profile;
	profile.neurons = std::vector<Neuron>(Compute::Random(1, maxNeurons));
	for (unsigned int i = 0; i < profile.neurons.size(); i++)
	{
		profile.neurons[i] = Neuron::Random(profile);
	}
	return profile;
}

void Network::Stream(unsigned int neuron)
{
	for (unsigned int x = 0; x < m_profile.neurons[neuron].connections.size(); x++)
	{
		Feed(m_profile.neurons[neuron].connections[x].connection, m_profile.neurons[neuron].Activate(m_profile.neurons[neuron].connections[x].bias + m_memory[neuron], m_profile.neurons[neuron].connections[x].weight));
	}
}

double Network::Read(unsigned int neuron)
{
	return m_memory[neuron];
}

void Network::Feed(unsigned int neuron, double input)
{
	m_memory[neuron] += Compute::Clamp(input, -1.0, 1.0);
}

Profile& Network::Breakdown()
{
	return m_profile;
}

double Compute::Clamp(double x, double a, double b)
{
	return x < min(a,b) ? min(a, b) : (x > max(a,b) ? max(a, b) : x);
}

double Compute::Range(double percentage, double a, double b)
{
	return min(a, b) + percentage * (max(a, b) - min(a, b));
}

int Compute::Random(int a, int b)
{
	srand(rand());
	return min(a, b) + (rand() % (max(a, b) - min(a, b) + 1));
}

double Compute::Random(double a, double b)
{
	srand(rand());
	return min(a, b) + (static_cast<double>(rand()) / RAND_MAX) * (max(a, b) - min(a, b));
}

double Compute::Sigmoid(double x)
{
	return (1 / (1 + exp(-x)));
}

double Compute::SigmoidDerivative(double x)
{
	return Sigmoid(x) * (1 - Sigmoid(x));
}

bool Compute::Mutate(double mutation)
{
	return Random(0.0f, 1.0f) <= abs(Clamp(mutation));
}

int Compute::Mutate(int a, unsigned int offset, double mutation)
{
	offset = Compute::Mutate(mutation) ? Random(0, offset) : 0;
	return a + (Compute::Mutate(0.5) ? offset : -((int)offset));
}

double Compute::Mutate(double a, double mutation)
{
	return Mutate(mutation) ? Random() : a;
}

Synaspe Compute::Mutate(Profile* profile, Synaspe synaspe, double mutation)
{
	return Synaspe(profile != nullptr ? Compute::Mutate(mutation) ? Compute::Random(0, profile->neurons.size()) : synaspe.connection : synaspe.connection, Mutate(synaspe.weight, mutation), Mutate(synaspe.bias, mutation));
}

Neuron Compute::Mutate(Profile* profile, Neuron neuron, double mutation)
{
	Neuron mutate;
	mutate.bias = Mutate(neuron.bias, mutation);
	unsigned int neurons = Mutate(neuron.connections.size(), 1, mutation);
	for (unsigned int i = 0; i < neurons; i++)
	{
		mutate.connections.push_back(i < neuron.connections.size() ?
			Mutate(profile, neuron.connections[i], mutation)
		    : Synaspe::Random(*profile));
	}
	return mutate;
}

Profile Compute::Mutate(Profile& profile, double mutation)
{
	Profile mutate;
	mutate.neurons = std::vector<Neuron>(Mutate(profile.neurons.size(), 1, mutation));
	for (unsigned int i = 0; i < mutate.neurons.size(); i++)
	{
		mutate.neurons[i] = i < profile.neurons.size() ?
			Mutate(&mutate, profile.neurons[i], mutation)
			: Neuron::Random(mutate);
	}
	return profile;
}

double Compute::Cross(double a, double b)
{
	return Mutate(0.5) ? a : b;
}

Synaspe Compute::Cross(Profile* profile, Synaspe a, Synaspe b)
{
	unsigned int connection = Mutate(0.5) ? a.connection : b.connection;
	connection =  profile != nullptr ?
		profile->Valid(connection) ? connection
		: profile->Valid(a.connection) ? a.connection
		: profile->Valid(b.connection) ? b.connection
		: Compute::Random(0, profile->neurons.size() - 1)
		: connection;
	return Synaspe(connection, Cross(a.weight, b.weight), Cross(a.bias, b.bias));
}

Neuron Compute::Cross(Profile* profile, Neuron a, Neuron b)
{
	std::vector<Synaspe> connections(Random((int)a.connections.size(), b.connections.size()));
	for (unsigned int i = 0; i < connections.size(); i++)
	{
		connections[i] = i < a.connections.size() ? 
			(i < b.connections.size() ? 
			Cross(profile, a.connections[i], b.connections[i]) 
			: a.connections[i]) 
			: b.connections[i];
	}
	return Neuron(Cross(a.bias, b.bias), connections);
}

Profile Compute::Cross(Profile& a, Profile& b)
{
	Profile profile;
	profile.neurons = std::vector<Neuron>(Compute::Random((int)a.neurons.size(), b.neurons.size()));
	for (unsigned int i = 0; i < profile.neurons.size(); i++)
	{
		profile.neurons[i] =
			i < a.neurons.size() ?
			(i < b.neurons.size() ?
					Cross(&profile, a.neurons[i], b.neurons[i])
				: a.neurons[i])
			: b.neurons[i];
	}
	return profile;
}

double sig_der(double x) 
{
	x = Compute::Sigmoid(x);
	return x * (1 - x);
}

void propagate(double& x, const double& propagation)
{
	x -= x - (x * sig_der(propagation));
}

void Compute::Configure(Profile* profile, unsigned int index, const double& delta)
{
	std::vector<Bridge> connections = profile->ConnectionsTo(index);
	double propagation = delta / (connections.size() + 1);
	propagate(profile->neurons[index].bias, propagation);
	for (unsigned int i = 0; i < connections.size(); i++)
	{
		Neuron& neuron = profile->neurons[connections[i].input];
		propagate(neuron.connections[connections[i].index].weight, propagation / 2);
		propagate(neuron.connections[connections[i].index].bias, propagation / 2);

		if (connections[i].input != index)
			propagate(profile->neurons[index].bias, propagation / (neuron.connections.size() + 1));
		else
			Configure(profile, connections[i].input, propagation);
	}
}

std::vector<Profile> Compute::CrossPopulation(std::vector<Profile> population, unsigned int size)
{
	std::vector<Profile> output;
	for (unsigned int i = 0; i < size; i++)
	{
		int mid = Compute::Random(0, population.size() - 1);
		output.push_back(Compute::Cross(population[Random(0, mid)], population[Random(mid, population.size() - 1)]));
	}
	return output;
}
