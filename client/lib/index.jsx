import React from 'react';
import ReactDOM from 'react-dom';
import $ from 'jquery';
import Chat from './Chat.jsx';
import ChatHistory from './ChatHistory.jsx';
import Waiting from './Waiting.jsx';
import Features from './Features.jsx';

const PREDICT_URL = 'http://localhost:8080/'

const heightMsg = 65;
const offsetBottomMsgs = 130;
let savedUserInput = "";

class App extends React.Component {
    fixFeaturesData(feats) {
      return feats.map( (d, i) => {
        return { data: d, i };
      });
    }
    constructor() {
        super();
        this.data = this.fixFeaturesData([0,0]);
        this.maxMsgs = 0;
        this.state = {
            endpts: { start: 0, end: this.maxMsgs },
            messages: [],
            features: [],
            bot: false
        };
    }

    componentDidMount() {
        this.updateMaxMsgs();
        this.bindEvents();
    }

    bindEvents() {
        let id;
        if (!this.CH) {
            this.CH = document.getElementById("all-messages");
            if (this.CH) {
                this.CH.addEventListener("mousewheel", (e) => {
                    if (!id) {
                        id = setTimeout(() => {
                            this.updateChatHistory.call(this, e);
                            id = null;
                        }, 10);
                    }
                });
            }
        }
    }

    updateChatHistory(e) {
        const { deltaY } = e;
        const { endpts, messages } = this.state;
        if (messages.length > this.maxMsgs) {
            const { start, end } = endpts;
            if (deltaY > 0) {
                this.setState({endpts: { start: Math.max(start-1, 0), end: Math.max(end-1, this.maxMsgs) }});
            }
            if (deltaY < 0 && deltaY !== -0) {
                this.setState({endpts:
                    { start: Math.min(start+1, messages.length - this.maxMsgs),
                      end: Math.min(end+1, messages.length) }
                });
            }
        }
    }

    updateMaxMsgs() {
        const heightWindow =  window.innerHeight - offsetBottomMsgs;
        this.maxMsgs = Math.floor(heightWindow/heightMsg);
    }

    addMessageToHistory(message, waitingForBot, featureProbs) {
        if (waitingForBot)
          featureProbs = [0,0];
        let { features, messages } = this.state;
        let newFeatures = new Features(null, 40, 500, this.fixFeaturesData(featureProbs));
        this.setState({
            endpts: { start: Math.max(messages.length-this.maxMsgs, 0), end: messages.length+1 },
            messages: messages.concat([message]),
            features: features.concat([newFeatures]),
            bot: waitingForBot
        });
    }

    post(userInput) {
        savedUserInput += " "+userInput
        this.addMessageToHistory({
            msg: userInput,
            user: "me",
            ts: new Date().toString()
        }, true);
        const url = PREDICT_URL + '?message=' + escape(userInput);
        $.get( url )
          .done((response) => {
            let data = JSON.parse(response);
            this.addMessageToHistory({
              msg: data.message,
              user: "bot",
              ts: new Date().toString()
            },
            false,
            data.predictions);
          })
          .fail(function(err) {
              console.log('error', err);
          });
    }

    getScrollableElements(msgs, feats, endpts) {
        const { start, end } = endpts;
        const newMessages = msgs.slice(start, end);
        const newFeatures = feats.slice(start, end);
        return { newFeatures, newMessages };
    }

    render() {
        const { bot, messages, endpts, features } = this.state;
        this.updateMaxMsgs();
        this.bindEvents();
        const { newMessages, newFeatures } = this.getScrollableElements(messages, features, endpts);
        return (
            <div>
                <ChatHistory features={ newFeatures } blurbs={ newMessages }/>
                { bot && <Waiting/> }
                <Chat waitingForBot={ bot } userInputRelayer={ this.post.bind(this) }/>
            </div>
         );
    }
}

ReactDOM.render(<App/>, document.getElementById("start"));
