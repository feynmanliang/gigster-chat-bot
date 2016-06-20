import React from 'react';
import ReactDOM from 'react-dom';

import Navbar from './Navbar.jsx';
import Header from './Header.jsx';
import Messages from './Messages.jsx';
import Footer from './Footer.jsx';

export default class App extends React.Component {
  render() {
    return (
      <div className="full height">
        <Navbar channel="Machine Sales Engineer" />
        <div className="main ui container">
          <Header channel="Machine Sales Engineer" />
          <MessageHistory />
          <Footer channel="Machine Sales Engineer" />
        </div>
      </div>
    );
  }
}

ReactDOM.render(<App />, document.getElementById('app'))
